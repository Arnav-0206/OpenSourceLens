import React from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ErrorStateCardProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  className?: string;
}

export default function ErrorStateCard({
  title = "Failed to load data",
  message = "An error occurred while connecting to the backend API. Please verify the service is running and try again.",
  onRetry,
  className,
}: ErrorStateCardProps) {
  return (
    <div
      className={cn(
        "p-6 rounded-2xl bg-rose-950/20 border border-rose-500/30 text-rose-200 backdrop-blur-sm space-y-4",
        className
      )}
    >
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center shrink-0">
          <AlertTriangle className="w-5 h-5 text-rose-400" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-rose-300">{title}</h3>
          <p className="text-xs text-rose-200/80 mt-1 leading-relaxed">
            {message}
          </p>
        </div>
      </div>

      {onRetry && (
        <div className="pt-2 border-t border-rose-500/20 flex justify-end">
          <button
            type="button"
            onClick={onRetry}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-medium text-xs transition-colors shadow-sm"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Try Again</span>
          </button>
        </div>
      )}
    </div>
  );
}
