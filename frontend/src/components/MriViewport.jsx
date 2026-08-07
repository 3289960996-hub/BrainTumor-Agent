import { useEffect, useRef } from "react";

import { grayscaleSlice, voxelValue } from "../nifti";

const COLORS = {
  wt: [34, 199, 214],
  tc: [245, 158, 11],
  et: [168, 85, 247],
};

function selectedColor(label, layers) {
  if (label === 4 && layers.et) {
    return COLORS.et;
  }
  if ((label === 1 || label === 4) && layers.tc) {
    return COLORS.tc;
  }
  if ((label === 1 || label === 2 || label === 4) && layers.wt) {
    return COLORS.wt;
  }
  return null;
}

export default function MriViewport({
  title,
  volume,
  mask,
  slice,
  opacity,
  layers,
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!volume || !canvasRef.current) {
      return;
    }
    const canvas = canvasRef.current;
    canvas.width = volume.width;
    canvas.height = volume.height;
    const context = canvas.getContext("2d");
    const pixels = grayscaleSlice(volume, slice);

    if (
      mask &&
      mask.width === volume.width &&
      mask.height === volume.height
    ) {
      const maskSlice = Math.min(mask.depth - 1, slice);
      const alpha = Math.min(0.85, Math.max(0, opacity));
      for (let y = 0; y < volume.height; y += 1) {
        const flippedY = volume.height - 1 - y;
        for (let x = 0; x < volume.width; x += 1) {
          const label = Math.round(voxelValue(mask, x, y, maskSlice));
          const color = selectedColor(label, layers);
          if (!color) {
            continue;
          }
          const index = (x + volume.width * flippedY) * 4;
          pixels[index] = Math.round(
            pixels[index] * (1 - alpha) + color[0] * alpha,
          );
          pixels[index + 1] = Math.round(
            pixels[index + 1] * (1 - alpha) + color[1] * alpha,
          );
          pixels[index + 2] = Math.round(
            pixels[index + 2] * (1 - alpha) + color[2] * alpha,
          );
        }
      }
    }

    context.putImageData(
      new ImageData(pixels, volume.width, volume.height),
      0,
      0,
    );
  }, [volume, mask, slice, opacity, layers]);

  return (
    <article className="mri-viewport">
      <span className="viewport-label">{title}</span>
      {volume ? (
        <canvas ref={canvasRef} aria-label={`${title} MRI切片`} />
      ) : (
        <div className="viewport-empty">
          <IconPlaceholder />
          <span>等待上传 {title}</span>
        </div>
      )}
    </article>
  );
}

function IconPlaceholder() {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height="34"
      viewBox="0 0 48 48"
      width="34"
    >
      <rect
        height="36"
        rx="6"
        stroke="currentColor"
        strokeWidth="2"
        width="36"
        x="6"
        y="6"
      />
      <path
        d="m14 31 7-7 5 5 4-4 5 6"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
      <circle cx="18" cy="17" fill="currentColor" r="3" />
    </svg>
  );
}
