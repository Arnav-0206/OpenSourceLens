import { Skeleton, SkeletonMetricsGrid } from "@/components/common/Skeleton";

export default function RepoDetailLoading() {
  return (
    <div className="space-y-8 pb-12 animate-pulse">
      {/* 1. RepoHeader Skeleton */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
        <div className="flex items-center gap-2">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-4 w-4" />
          <Skeleton className="h-4 w-32" />
        </div>
        <div className="flex items-center justify-between">
          <Skeleton className="h-9 w-64" />
          <div className="flex items-center gap-2">
            <Skeleton className="h-7 w-20 rounded-lg" />
            <Skeleton className="h-7 w-20 rounded-lg" />
          </div>
        </div>
        <Skeleton className="h-4 w-3/4 max-w-xl" />
      </div>

      {/* 2. Metrics Grid Skeleton */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <Skeleton className="h-4 w-36" />
          <Skeleton className="h-4 w-24" />
        </div>
        <SkeletonMetricsGrid count={5} />
      </section>

      {/* 3. Trends Chart Skeleton */}
      <section className="h-80 rounded-2xl bg-slate-900/60 border border-slate-800 p-6 space-y-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-5 w-48" />
          <Skeleton className="h-5 w-32" />
        </div>
        <Skeleton className="h-56 w-full rounded-xl" />
      </section>

      {/* 4. Two Distribution Cards Skeleton */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="h-72 rounded-2xl bg-slate-900/60 border border-slate-800 p-6 space-y-4">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-48 w-full rounded-xl" />
        </div>
        <div className="h-72 rounded-2xl bg-slate-900/60 border border-slate-800 p-6 space-y-4">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-48 w-full rounded-xl" />
        </div>
      </section>
    </div>
  );
}
