import assert from "node:assert/strict";
import test from "node:test";
import { gzipSync } from "node:zlib";

import {
  findTumorSlice,
  grayscaleSlice,
  parseNiftiBuffer,
  voxelValue,
} from "./nifti.js";

function niftiUint8(width, height, depth, values) {
  const dataOffset = 352;
  const buffer = new ArrayBuffer(dataOffset + width * height * depth);
  const view = new DataView(buffer);
  view.setInt32(0, 348, true);
  view.setInt16(40, 3, true);
  view.setInt16(42, width, true);
  view.setInt16(44, height, true);
  view.setInt16(46, depth, true);
  view.setInt16(70, 2, true);
  view.setInt16(72, 8, true);
  view.setFloat32(108, dataOffset, true);
  new Uint8Array(buffer, dataOffset).set(values);
  return buffer;
}

test("解析NIfTI尺寸和体素", async () => {
  const buffer = niftiUint8(2, 2, 2, [0, 10, 20, 30, 40, 50, 60, 70]);
  const volume = await parseNiftiBuffer(buffer);

  assert.equal(volume.width, 2);
  assert.equal(volume.height, 2);
  assert.equal(volume.depth, 2);
  assert.equal(voxelValue(volume, 1, 1, 1), 70);
  assert.equal(grayscaleSlice(volume, 1).length, 16);
});

test("解析gzip压缩NIfTI并定位肿瘤切片", async () => {
  const labels = new Uint8Array(3 * 3 * 3);
  labels[1 + 3 * (1 + 3 * 2)] = 4;
  const source = niftiUint8(3, 3, 3, labels);
  const compressed = gzipSync(Buffer.from(source));
  const compressedBuffer = compressed.buffer.slice(
    compressed.byteOffset,
    compressed.byteOffset + compressed.byteLength,
  );
  const volume = await parseNiftiBuffer(compressedBuffer);

  assert.equal(findTumorSlice(volume), 2);
});
