import assert from "node:assert/strict";
import test from "node:test";
import { gzipSync } from "node:zlib";

import {
  findTumorSlice,
  grayscaleSlice,
  parseNiftiBuffer,
  sliceTumorMetrics,
  voxelValue,
} from "./nifti.js";

function niftiUint8(width, height, depth, values, spacing = [1, 1, 1]) {
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
  view.setFloat32(80, spacing[0], true);
  view.setFloat32(84, spacing[1], true);
  view.setFloat32(88, spacing[2], true);
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
  assert.deepEqual(volume.spacing, { x: 1, y: 1, z: 1 });
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

test("当前切片病灶面积和最大径随Mask变化", async () => {
  const labels = new Uint8Array(4 * 3 * 2);
  const sliceOffset = 4 * 3;
  labels[sliceOffset] = 2;
  labels[sliceOffset + 1] = 1;
  labels[sliceOffset + 2] = 4;
  const volume = await parseNiftiBuffer(
    niftiUint8(4, 3, 2, labels, [2, 3, 4]),
  );

  const empty = sliceTumorMetrics(volume, 0);
  const lesion = sliceTumorMetrics(volume, 1);

  assert.equal(empty.wholeTumorArea, 0);
  assert.equal(lesion.wholeTumorArea, 0.18);
  assert.equal(lesion.tumorCoreArea, 0.12);
  assert.equal(lesion.enhancingTumorArea, 0.06);
  assert.equal(lesion.maximumDiameter, 4);
});
