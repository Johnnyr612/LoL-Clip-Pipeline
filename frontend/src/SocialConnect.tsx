import React from "react";

type TikTokStatus = {
  connected: boolean;
  oauth_configured: boolean;
};

export function SocialConnect() {
  const [status, setStatus] = React.useState<TikTokStatus | null>(null);

  React.useEffect(() => {
    async function loadStatus() {
      const next = await fetch("/auth/tiktok/status").then((response) => (response.ok ? response.json() : null));
      setStatus(next);
    }

    void loadStatus();
    const timer = window.setInterval(loadStatus, 10000);
    return () => window.clearInterval(timer);
  }, []);

  function connectTikTok() {
    if (!status?.oauth_configured) return;
    window.open("/auth/tiktok/start", "_blank", "noopener,noreferrer");
  }

  const canConnect = Boolean(status?.oauth_configured && !status.connected);

  return (
    <div className="flex items-center gap-2 text-sm">
      <button
        className={status?.connected ? "border border-success bg-white px-3 py-2 font-medium text-success" : "border border-lane bg-white px-3 py-2 font-medium disabled:opacity-50"}
        disabled={!status?.connected && !canConnect}
        onClick={connectTikTok}
        title={!status?.connected && !status?.oauth_configured ? "Set TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET to enable TikTok OAuth." : undefined}
      >
        {status?.connected ? "TikTok connected" : status?.oauth_configured ? "Connect TikTok" : "Configure TikTok"}
      </button>
      <button className="border border-lane bg-white px-3 py-2 font-medium">Instagram</button>
    </div>
  );
}
