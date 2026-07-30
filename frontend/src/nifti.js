const DATATYPE_READERS = {
  2: { bytes: 1, read: (view, offset) => view.getUint8(offset) },
  4: {
    bytes: 2,
    read: (view, offset, littleEndian) =>
      view.getInt16(offset, littleEndian),
  },
  8: {
    bytes: 4,
    read: (view, offset, littleEndian) =>
      view.getInt32(offset, littleEndian),
  },
  16: {
    bytes: 4,
    read: (view, offset, littleEndian) =>
      view.getFloat32(offset, littleEndian),
  },
  64: {
    bytes: 8,
    read: (view, offset, littleEndian) =>
      view.getFloat64(offset, littleEndian),
  },
  256: { bytes: 1, read: (view, offset) => view.getInt8(offset) },
  512: {
    bytes: 2,
    read: (view, offset, littleEndian) =>
      view.getUint16(offset, littleEndian),
  },
  768: {
    bytes: 4,
    read: (view, offset, littleEndian) =>
      view.getUint32(offset, littleEndian),
  },
};

async function decompressIfNeeded(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) {
    return buffer;
  }
  if (typeof DecompressionStream === "undefined") {
    throw new Error("当前浏览器不支持gzip解压，请使用最新版Chrome或Edge");
  }
  const decompressed = new Blob([buffer])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"));
  return new Response(decompressed).arrayBuffer();
}

export async function parseNiftiBuffer(sourceBuffer) {
  const buffer = await decompressIfNeeded(sourceBuffer);
  const view = new DataView(buffer);
  const littleHeader = view.getInt32(0, true);
  const bigHeader = view.getInt32(0, false);
  if (littleHeader !== 348 && bigHeader !== 348) {
    throw new Error("文件不是有效的NIfTI-1格式");
  }

  const littleEndian = littleHeader === 348;
  const dimensions = view.getInt16(40, littleEndian);
  if (dimensions < 3) {
    throw new Error("仅支持三维NIfTI影像");
  }
  const width = view.getInt16(42, littleEndian);
  const height = view.getInt16(44, littleEndian);
  const depth = view.getInt16(46, littleEndian);
  if (width < 1 || height < 1 || depth < 1) {
    throw new Error("NIfTI尺寸无效");
  }

  const datatype = view.getInt16(70, littleEndian);
  const reader = DATATYPE_READERS[datatype];
  if (!reader) {
    throw new Error(`暂不支持NIfTI datatype=${datatype}`);
  }
  const dataOffset = Math.max(
    352,
    Math.floor(view.getFloat32(108, littleEndian)),
  );
  const slopeValue = view.getFloat32(112, littleEndian);
  const interceptValue = view.getFloat32(116, littleEndian);
  const slope = Number.isFinite(slopeValue) && slopeValue !== 0 ? slopeValue : 1;
  const intercept = Number.isFinite(interceptValue) ? interceptValue : 0;
  const voxelCount = width * height * depth;
  if (dataOffset + voxelCount * reader.bytes > buffer.byteLength) {
    throw new Error("NIfTI体素数据不完整");
  }

  return {
    width,
    height,
    depth,
    datatype,
    littleEndian,
    dataOffset,
    slope,
    intercept,
    reader,
    view,
  };
}

export async function parseNiftiFile(file) {
  return parseNiftiBuffer(await file.arrayBuffer());
}

export function voxelValue(volume, x, y, z) {
  const index = x + volume.width * (y + volume.height * z);
  const raw = volume.reader.read(
    volume.view,
    volume.dataOffset + index * volume.reader.bytes,
    volume.littleEndian,
  );
  return raw * volume.slope + volume.intercept;
}

function percentile(sorted, ratio) {
  if (!sorted.length) {
    return 0;
  }
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.floor((sorted.length - 1) * ratio)),
  );
  return sorted[index];
}

export function grayscaleSlice(volume, sliceIndex) {
  const z = Math.min(volume.depth - 1, Math.max(0, sliceIndex));
  const values = [];
  const slice = new Float32Array(volume.width * volume.height);

  for (let y = 0; y < volume.height; y += 1) {
    for (let x = 0; x < volume.width; x += 1) {
      const value = voxelValue(volume, x, y, z);
      const index = x + volume.width * y;
      slice[index] = value;
      if (Number.isFinite(value) && value !== 0) {
        values.push(value);
      }
    }
  }

  values.sort((a, b) => a - b);
  const low = percentile(values, 0.01);
  const high = percentile(values, 0.995);
  const range = high > low ? high - low : 1;
  const pixels = new Uint8ClampedArray(volume.width * volume.height * 4);

  for (let y = 0; y < volume.height; y += 1) {
    const flippedY = volume.height - 1 - y;
    for (let x = 0; x < volume.width; x += 1) {
      const sourceIndex = x + volume.width * y;
      const targetIndex = (x + volume.width * flippedY) * 4;
      const normalized = Math.min(
        1,
        Math.max(0, (slice[sourceIndex] - low) / range),
      );
      const intensity = Math.round(255 * normalized ** 0.82);
      pixels[targetIndex] = intensity;
      pixels[targetIndex + 1] = intensity;
      pixels[targetIndex + 2] = intensity;
      pixels[targetIndex + 3] = 255;
    }
  }
  return pixels;
}

export function findTumorSlice(mask) {
  const counts = new Uint32Array(mask.depth);
  for (let z = 0; z < mask.depth; z += 1) {
    let count = 0;
    for (let y = 0; y < mask.height; y += 1) {
      for (let x = 0; x < mask.width; x += 1) {
        if (Math.round(voxelValue(mask, x, y, z)) > 0) {
          count += 1;
        }
      }
    }
    counts[z] = count;
  }
  let bestSlice = Math.floor(mask.depth / 2);
  let bestCount = 0;
  counts.forEach((count, index) => {
    if (count > bestCount) {
      bestCount = count;
      bestSlice = index;
    }
  });
  return bestSlice;
}
