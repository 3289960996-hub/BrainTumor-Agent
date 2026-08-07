import { useMemo, useState } from "react";

import Icon from "./Icon";
import {
  autoMatchModalities,
  MODALITIES,
  validateModalityAssignments,
} from "../modalityFiles";

export default function UploadDialog({ open, busy, onClose, onSubmit }) {
  const [files, setFiles] = useState({});
  const [caseId, setCaseId] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [matchFeedback, setMatchFeedback] = useState(null);
  const complete = useMemo(
    () => MODALITIES.every(([key]) => files[key]) && confirmed,
    [confirmed, files],
  );

  if (!open) {
    return null;
  }

  function updateFile(modality, file) {
    setFiles((current) => ({ ...current, [modality]: file }));
    setConfirmed(false);
    setMatchFeedback(null);
  }

  function matchSelectedFiles(event) {
    try {
      const matched = autoMatchModalities(event.target.files);
      setFiles(matched);
      setConfirmed(false);
      setMatchFeedback({
        type: "success",
        message: "已按文件名匹配四个模态，请逐项核对后确认。",
      });
    } catch (error) {
      setMatchFeedback({ type: "error", message: error.message });
    } finally {
      event.target.value = "";
    }
  }

  async function submit(event) {
    event.preventDefault();
    if (!complete || busy) {
      return;
    }
    try {
      validateModalityAssignments(files);
    } catch (error) {
      setConfirmed(false);
      setMatchFeedback({ type: "error", message: error.message });
      return;
    }
    const uploaded = await onSubmit(files, caseId);
    if (uploaded) {
      setFiles({});
      setCaseId("");
      setConfirmed(false);
      setMatchFeedback(null);
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
          <label className="auto-match-field">
            <Icon name="upload" />
            <span>
              <strong>自动匹配四模态</strong>
              <small>一次选择4个文件，按文件名识别T1、T1ce、T2和FLAIR</small>
            </span>
            <input
              accept=".nii,.nii.gz,application/gzip"
              multiple
              onChange={matchSelectedFiles}
              type="file"
            />
          </label>
          {matchFeedback && (
            <p className={`modality-match-feedback ${matchFeedback.type}`} role="status">
              {matchFeedback.message}
            </p>
          )}

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

          <label className="modality-confirmation">
            <input
              checked={confirmed}
              disabled={!MODALITIES.every(([key]) => files[key]) || busy}
              onChange={(event) => setConfirmed(event.target.checked)}
              type="checkbox"
            />
            <span>我已核对四个MRI模态与文件对应正确</span>
          </label>

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
