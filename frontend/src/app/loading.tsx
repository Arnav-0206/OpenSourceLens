import { Skeleton, SkeletonMetricsGrid } from "@/components/common/Skeleton";

export default function RootLoading() {
  return (
    <div className="space-y-8 pb-12 animate-pulse">
      {/* Header skeleton */}
      <div className="space-y-3">
        <Skeleton className="h-5 w-40 rounded-full" />
        <Skeleton className="h-10 w-96 max-w-full" />
        <Skeleton className="h-4 w-2/3 max-w-xl" />
      </div>

      {/* Metrics Grid Skeleton */}
      <SkeletonMetricsGrid count={4} />

      {/* Main Content Panels Skeleton */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-4">
        <div className="h-72 rounded-2xl bg-slate-900/60 border border-slate-800 p-6 space-y-4">
          <Skeleton className="h-5 w-44" />
          <Skeleton className="h-44 w-full rounded-xl" />
        </div>
        <div className="h-72 rounded-2xl bg-slate-900/60 border border-slate-800 p-6 space-y-4">
          <Skeleton className="h-5 w-44" />
          <Skeleton className="h-44 w-full rounded-xl" />
        </div>
      </div>
    </div>
  );
}
