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
  const spacing = {
    x: positiveSpacing(view.getFloat32(80, littleEndian)),
    y: positiveSpacing(view.getFloat32(84, littleEndian)),
    z: positiveSpacing(view.getFloat32(88, littleEndian)),
  };
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
    spacing,
    reader,
    view,
  };
}

function positiveSpacing(value) {
  return Number.isFinite(value) && value !== 0 ? Math.abs(value) : 1;
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

function isWholeTumor(label) {
  return label === 1 || label === 2 || label === 4;
}

function cross(origin, first, second) {
  return (
    (first.x - origin.x) * (second.y - origin.y) -
    (first.y - origin.y) * (second.x - origin.x)
  );
}

function convexHull(points) {
  if (points.length <= 2) {
    return points;
  }
  const sorted = [...points].sort((first, second) =>
    first.x === second.x ? first.y - second.y : first.x - second.x,
  );
  const lower = [];
  for (const point of sorted) {
    while (
      lower.length >= 2 &&
      cross(lower.at(-2), lower.at(-1), point) <= 0
    ) {
      lower.pop();
    }
    lower.push(point);
  }
  const upper = [];
  for (const point of sorted.toReversed()) {
    while (
      upper.length >= 2 &&
      cross(upper.at(-2), upper.at(-1), point) <= 0
    ) {
      upper.pop();
    }
    upper.push(point);
  }
  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function maximumDiameter(points) {
  const hull = convexHull(points);
  let maximumSquared = 0;
  for (let first = 0; first < hull.length; first += 1) {
    for (let second = first + 1; second < hull.length; second += 1) {
      const deltaX = hull[first].x - hull[second].x;
      const deltaY = hull[first].y - hull[second].y;
      maximumSquared = Math.max(
        maximumSquared,
        deltaX * deltaX + deltaY * deltaY,
      );
    }
  }
  return Math.sqrt(maximumSquared);
}

export function sliceTumorMetrics(mask, sliceIndex) {
  if (!mask) {
    return null;
  }
  const z = Math.min(mask.depth - 1, Math.max(0, sliceIndex));
  const labels = new Uint8Array(mask.width * mask.height);
  const counts = { wholeTumor: 0, tumorCore: 0, enhancingTumor: 0 };

  for (let y = 0; y < mask.height; y += 1) {
    for (let x = 0; x < mask.width; x += 1) {
      const label = Math.round(voxelValue(mask, x, y, z));
      labels[x + mask.width * y] = label;
      if (isWholeTumor(label)) {
        counts.wholeTumor += 1;
      }
      if (label === 1 || label === 4) {
        counts.tumorCore += 1;
      }
      if (label === 4) {
        counts.enhancingTumor += 1;
      }
    }
  }

  const boundary = [];
  for (let y = 0; y < mask.height; y += 1) {
    for (let x = 0; x < mask.width; x += 1) {
      const index = x + mask.width * y;
      if (!isWholeTumor(labels[index])) {
        continue;
      }
      const isBoundary =
        x === 0 ||
        y === 0 ||
        x === mask.width - 1 ||
        y === mask.height - 1 ||
        !isWholeTumor(labels[index - 1]) ||
        !isWholeTumor(labels[index + 1]) ||
        !isWholeTumor(labels[index - mask.width]) ||
        !isWholeTumor(labels[index + mask.width]);
      if (isBoundary) {
        boundary.push({
          x: x * mask.spacing.x,
          y: y * mask.spacing.y,
        });
      }
    }
  }

  const voxelAreaCm2 = (mask.spacing.x * mask.spacing.y) / 100;
  return {
    slice: z,
    wholeTumorArea: counts.wholeTumor * voxelAreaCm2,
    tumorCoreArea: counts.tumorCore * voxelAreaCm2,
    enhancingTumorArea: counts.enhancingTumor * voxelAreaCm2,
    maximumDiameter: maximumDiameter(boundary),
  };
}
