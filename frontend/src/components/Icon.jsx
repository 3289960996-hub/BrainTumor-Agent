const paths = {
  home: (
    <>
      <path d="m3 11 9-8 9 8" />
      <path d="M5 10v10h14V10M9 20v-6h6v6" />
    </>
  ),
  upload: (
    <>
      <path d="M12 16V4m0 0L7 9m5-5 5 5" />
      <path d="M5 14H3v6h18v-6h-2" />
    </>
  ),
  brain: (
    <>
      <path d="M9.5 4.5A3.5 3.5 0 0 0 6 8v1a3 3 0 0 0-2 5.6A3.5 3.5 0 0 0 9.5 19" />
      <path d="M14.5 4.5A3.5 3.5 0 0 1 18 8v1a3 3 0 0 1 2 5.6 3.5 3.5 0 0 1-5.5 4.4M12 3v18M8 9h4m0 6h4" />
    </>
  ),
  monitor: (
    <>
      <rect x="3" y="4" width="18" height="13" rx="2" />
      <path d="M8 21h8m-4-4v4" />
    </>
  ),
  report: (
    <>
      <path d="M6 2h9l4 4v16H6z" />
      <path d="M14 2v5h5M9 12h6m-6 4h6" />
    </>
  ),
  chat: (
    <>
      <path d="M21 12a8 8 0 0 1-8 8H7l-4 2 1.2-4A9 9 0 1 1 21 12Z" />
      <path d="M8 12h.01M12 12h.01M16 12h.01" />
    </>
  ),
  folder: (
    <path d="M3 6h7l2 2h9v11H3zM3 6V4h7l2 2" />
  ),
  download: (
    <>
      <path d="M12 3v12m0 0-5-5m5 5 5-5" />
      <path d="M4 19v2h16v-2" />
    </>
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19 13.5v-3l-2-.7-.8-1.8.9-1.9L15 4l-1.9.9-1.9-.8L10.5 2h-3l-.7 2.1-1.8.8L3.1 4 1 6.1 1.9 8 1.1 9.8 0 10.5v3l2.1.7.8 1.8-.9 1.9L4.1 20l1.9-.9 1.8.8.7 2.1h3l.7-2.1 1.8-.8 1.9.9 2.1-2.1-.9-1.9.8-1.8Z" transform="translate(2 -1)" />
    </>
  ),
  shield: (
    <>
      <path d="M12 2 4 5v6c0 5 3.4 8.8 8 11 4.6-2.2 8-6 8-11V5z" />
      <path d="m9 12 2 2 4-5" />
    </>
  ),
  check: <path d="m5 12 4 4L19 6" />,
  layers: (
    <>
      <path d="m12 2 9 5-9 5-9-5z" />
      <path d="m3 12 9 5 9-5M3 17l9 5 9-5" />
    </>
  ),
  source: (
    <>
      <path d="M4 4h16v16H4z" />
      <path d="M8 8h8m-8 4h8m-8 4h5" />
    </>
  ),
  send: (
    <>
      <path d="m3 3 18 9-18 9 3-9z" />
      <path d="M6 12h15" />
    </>
  ),
  close: (
    <>
      <path d="m6 6 12 12M18 6 6 18" />
    </>
  ),
};

export default function Icon({ name, size = 20, strokeWidth = 1.8 }) {
  return (
    <svg
      aria-hidden="true"
      className="icon"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={strokeWidth}
    >
      {paths[name] || paths.source}
    </svg>
  );
}
