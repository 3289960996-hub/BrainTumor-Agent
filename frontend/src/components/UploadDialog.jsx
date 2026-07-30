import { useMemo, useState } from "react";

import Icon from "./Icon";

const MODALITIES = [
  ["t1", "T1"],
  ["t1ce", "T1ce"],
  ["t2", "T2"],
  ["flair", "FLAIR"],
];

export default function UploadDialog({ open, busy, onClose, onSubmit }) {
  const [files, setFiles] = useState({});
  const [caseId, setCaseId] = useState("");
  const complete = useMemo(
    () => MODALITIES.every(([key]) => files[key]),
    [files],
  );

  if (!open) {
    return null;
  }

  function updateFile(modality, file) {
    setFiles((current) => ({ ...current, [modality]: file }));
  }

  async function submit(event) {
    event.preventDefault();
    if (!complete || busy) {
      return;
    }
    const uploaded = await onSubmit(files, caseId);
    if (uploaded) {
      setFiles({});
      setCaseId("");
    }
  }

  return (
    <div className="dialog-backdrop" role="presentation">
      <section
        aria-labelledby="upload-title"
        aria-modal="true"
        className="upload-dialog"
        role="dialog"
      >
        <header>
          <div>
            <span className="dialog-kicker">BraTS 2021 · NIfTI</span>
            <h2 id="upload-title">上传四模态MRI</h2>
          </div>
          <button
            aria-label="关闭上传窗口"
            className="icon-button"
            disabled={busy}
            onClick={onClose}
            type="button"
          >
            <Icon name="close" />
          </button>
        </header>

        <form onSubmit={submit}>
          <label className="case-id-field">
            <span>病例编号（可选）</span>
            <input
              maxLength="64"
              onChange={(event) => setCaseId(event.target.value)}
              pattern="[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
              placeholder="留空将由服务端生成"
              value={caseId}
            />
            <small>请使用去标识化编号，不要填写姓名或住院号。</small>
          </label>

          <div className="upload-grid">
            {MODALITIES.map(([key, label]) => (
              <label className={files[key] ? "file-field selected" : "file-field"} key={key}>
                <Icon name={files[key] ? "check" : "upload"} />
                <strong>{label}</strong>
                <span>{files[key]?.name || "选择 .nii 或 .nii.gz"}</span>
                <input
                  accept=".nii,.nii.gz,application/gzip"
                  onChange={(event) => updateFile(key, event.target.files[0])}
                  type="file"
                />
              </label>
            ))}
          </div>

          <footer>
            <span className="upload-progress">
              已选择 {Object.keys(files).filter((key) => files[key]).length} / 4
            </span>
            <button
              className="secondary-button"
              disabled={busy}
              onClick={onClose}
              type="button"
            >
              取消
            </button>
            <button
              className="primary-button"
              disabled={!complete || busy}
              type="submit"
            >
              <Icon name="upload" />
              {busy ? "正在上传…" : "上传MRI"}
            </button>
          </footer>
        </form>
      </section>
    </div>
  );
}
