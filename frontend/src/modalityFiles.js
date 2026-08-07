export const MODALITIES = [
  ["t1", "T1"],
  ["t1ce", "T1ce"],
  ["t2", "T2"],
  ["flair", "FLAIR"],
];

const MODALITY_LABELS = Object.fromEntries(MODALITIES);

function filenameStem(filename) {
  return String(filename || "")
    .toLowerCase()
    .replace(/\.nii(?:\.gz)?$/, "")
    .replace(/[^a-z0-9]+/g, "_");
}

function hasToken(stem, pattern) {
  return new RegExp(`(?:^|_)${pattern}(?:_|$)`).test(stem);
}

export function inferModality(filename) {
  const stem = filenameStem(filename);
  if (hasToken(stem, "flair")) {
    return "flair";
  }
  if (
    hasToken(stem, "t1(?:ce|c|gd|post)") ||
    /(?:^|_)t1_(?:ce|c|gd|post)(?:_|$)/.test(stem)
  ) {
    return "t1ce";
  }
  if (hasToken(stem, "t2")) {
    return "t2";
  }
  if (hasToken(stem, "t1")) {
    return "t1";
  }
  return null;
}

export function autoMatchModalities(fileList) {
  const selected = Array.from(fileList || []);
  if (selected.length !== MODALITIES.length) {
    throw new Error("请一次选择T1、T1ce、T2和FLAIR共4个文件");
  }

  const matched = {};
  for (const file of selected) {
    const modality = inferModality(file.name);
    if (!modality) {
      throw new Error(`无法从文件名识别模态：${file.name}`);
    }
    if (matched[modality]) {
      throw new Error(
        `${MODALITY_LABELS[modality]}存在重复文件：${matched[modality].name}、${file.name}`,
      );
    }
    matched[modality] = file;
  }

  const missing = MODALITIES.filter(([key]) => !matched[key]).map(([, label]) => label);
  if (missing.length) {
    throw new Error(`缺少模态：${missing.join("、")}`);
  }
  return matched;
}

export function validateModalityAssignments(files) {
  const seen = new Set();
  for (const [modality, label] of MODALITIES) {
    const file = files[modality];
    if (!file) {
      throw new Error(`缺少${label}文件`);
    }
    if (seen.has(file)) {
      throw new Error("同一个文件不能用于多个MRI模态");
    }
    seen.add(file);
    const inferred = inferModality(file.name);
    if (inferred && inferred !== modality) {
      throw new Error(
        `${file.name}的文件名更像${MODALITY_LABELS[inferred]}，请检查${label}上传位置`,
      );
    }
  }
  return true;
}
