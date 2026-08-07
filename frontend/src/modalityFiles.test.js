import assert from "node:assert/strict";
import test from "node:test";

import {
  autoMatchModalities,
  inferModality,
  validateModalityAssignments,
} from "./modalityFiles.js";

function file(name) {
  return { name };
}

test("从常见BraTS文件名识别四模态", () => {
  assert.equal(inferModality("BraTS_001_t1.nii.gz"), "t1");
  assert.equal(inferModality("BraTS_001_t1ce.nii.gz"), "t1ce");
  assert.equal(inferModality("case-T1_GD.nii"), "t1ce");
  assert.equal(inferModality("BraTS_001_t2.nii.gz"), "t2");
  assert.equal(inferModality("BraTS_001_flair.nii.gz"), "flair");
  assert.equal(inferModality("scan.nii.gz"), null);
});

test("批量选择四文件后自动匹配", () => {
  const matched = autoMatchModalities([
    file("case_flair.nii.gz"),
    file("case_t2.nii.gz"),
    file("case_t1ce.nii.gz"),
    file("case_t1.nii.gz"),
  ]);

  assert.equal(matched.t1.name, "case_t1.nii.gz");
  assert.equal(matched.t1ce.name, "case_t1ce.nii.gz");
  assert.equal(matched.t2.name, "case_t2.nii.gz");
  assert.equal(matched.flair.name, "case_flair.nii.gz");
});

test("自动匹配拒绝未知和重复模态", () => {
  assert.throws(
    () => autoMatchModalities([
      file("case_t1.nii.gz"),
      file("case_t1ce.nii.gz"),
      file("case_t2.nii.gz"),
      file("scan.nii.gz"),
    ]),
    /无法从文件名识别模态/,
  );
  assert.throws(
    () => autoMatchModalities([
      file("a_t1.nii.gz"),
      file("b_t1.nii.gz"),
      file("case_t2.nii.gz"),
      file("case_flair.nii.gz"),
    ]),
    /存在重复文件/,
  );
});

test("手动上传时拒绝明显放错槽位", () => {
  assert.throws(
    () => validateModalityAssignments({
      t1: file("case_t2.nii.gz"),
      t1ce: file("case_t1ce.nii.gz"),
      t2: file("case_t1.nii.gz"),
      flair: file("case_flair.nii.gz"),
    }),
    /请检查T1上传位置/,
  );
});
