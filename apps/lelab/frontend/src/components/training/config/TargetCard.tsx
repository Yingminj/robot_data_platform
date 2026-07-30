import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ConfigComponentProps } from "../types";
import { ClusterNode, RunnerFlavor } from "@/lib/jobsApi";

interface TargetCardProps extends ConfigComponentProps {
  authenticated: boolean;
  flavors: RunnerFlavor[];
  loading: boolean;
  clusterEnabled: boolean;
  clusterNodes: ClusterNode[];
}

const formatHourly = (unitCostUsd: number, unitLabel: string): string => {
  const hourly = unitLabel === "minute" ? unitCostUsd * 60 : unitCostUsd;
  return `$${hourly.toFixed(2)}/hr`;
};

const formatFlavorLine = (f: RunnerFlavor): string => {
  const accel = f.accelerator ? f.accelerator : f.cpu;
  return `${f.pretty_name} · ${accel} · ${formatHourly(f.unit_cost_usd, f.unit_label)}`;
};

const TargetCard: React.FC<TargetCardProps> = ({
  config,
  updateConfig,
  authenticated,
  flavors,
  loading,
  clusterEnabled,
  clusterNodes,
}) => {
  const target = config.target;
  const value = target.runner === "local"
    ? "local"
    : target.runner === "slurm"
      ? `slurm:${target.node ?? "auto"}`
      : `hf:${target.flavor ?? ""}`;

  const handleChange = (v: string) => {
    if (v === "local") {
      updateConfig("target", { runner: "local" });
    } else if (v.startsWith("slurm:")) {
      updateConfig("target", {
        runner: "slurm",
        node: v.slice("slurm:".length),
      });
    } else if (v.startsWith("hf:")) {
      const flavor = v.slice("hf:".length);
      updateConfig("target", { runner: "hf_cloud", flavor });
    }
  };

  return (
    <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
      <CardHeader>
        <CardTitle className="text-white">Compute target</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <Label className="text-slate-300">Run training on</Label>
          <Select value={value} onValueChange={handleChange}>
            <SelectTrigger className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1">
              <SelectValue placeholder={loading ? "Loading…" : "Select target"} />
            </SelectTrigger>
            <SelectContent className="bg-slate-800 border-slate-600 text-white">
              <SelectItem value="slurm:auto" disabled={!clusterEnabled}>
                Cluster — automatically select a free GPU
              </SelectItem>
              {clusterNodes.map((node) => (
                <SelectItem
                  key={node.name}
                  value={`slurm:${node.name}`}
                  disabled={!node.eligible}
                >
                  {node.name} · {node.gpu_name ?? "GPU unavailable"} ·{" "}
                  {node.memory_free_mb == null
                    ? "unknown memory"
                    : `${Math.round(node.memory_free_mb / 1024)} GiB free`}
                </SelectItem>
              ))}
              <SelectItem value="local">Local — your machine (free)</SelectItem>
              {flavors.map((f) => (
                <SelectItem
                  key={f.name}
                  value={`hf:${f.name}`}
                  disabled={!authenticated}
                >
                  {formatFlavorLine(f)}
                  {!authenticated && (
                    <span className="text-amber-300 ml-2 text-xs">
                      log in to HF
                    </span>
                  )}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {target.runner === "slurm" ? (
            <div className="mt-3 grid gap-2">
              {clusterNodes.map((node) => (
                <div
                  key={node.name}
                  className="flex items-center justify-between rounded-md border border-slate-700 bg-slate-900/60 px-3 py-2 text-xs"
                >
                  <span className="text-slate-200">
                    {node.name} · {node.gpu_name ?? node.address}
                  </span>
                  <span className={node.eligible ? "text-green-300" : "text-amber-300"}>
                    {node.eligible ? "GPU idle" : node.reason ?? node.slurm_state}
                  </span>
                </div>
              ))}
              {!clusterEnabled && !loading && (
                <p className="text-amber-300">
                  Cluster runner is disabled on the management server.
                </p>
              )}
            </div>
          ) : (
            <p className="text-xs text-slate-500 mt-1">
              Cloud cost is shown per running hour. Local uses this Web server.
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default TargetCard;
