import { Skeleton, SkeletonMetricsGrid } from "@/components/common/Skeleton";

export default function AnalyticsLoading() {
  return (
    <div className="space-y-8 pb-12 animate-pulse">
      {/* Header */}
      <div className="space-y-3">
        <Skeleton className="h-5 w-44 rounded-full" />
        <Skeleton className="h-10 w-96 max-w-full" />
        <Skeleton className="h-4 w-2/3 max-w-xl" />
      </div>

      {/* 1. Global Metrics Skeleton */}
      <section className="space-y-3">
        <Skeleton className="h-4 w-40" />
        <SkeletonMetricsGrid count={4} />
      </section>

      {/* 2. Visual Charts Skeleton Grid */}
      <section className="space-y-3">
        <Skeleton className="h-4 w-48" />
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          <div className="lg:col-span-7 h-72 rounded-2xl bg-slate-900/60 border border-slate-800 p-6 space-y-4">
            <Skeleton className="h-5 w-48" />
            <Skeleton className="h-48 w-full rounded-xl" />
          </div>
          <div className="lg:col-span-5 h-72 rounded-2xl bg-slate-900/60 border border-slate-800 p-6 space-y-4">
            <Skeleton className="h-5 w-48" />
            <Skeleton className="h-48 w-full rounded-xl" />
          </div>
        </div>
      </section>

      {/* 3. Comparative Matrix Table Skeleton */}
      <section className="space-y-3">
        <Skeleton className="h-4 w-40" />
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 space-y-4">
          <div className="h-10 w-full bg-slate-950/60 rounded-xl" />
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className="h-14 w-full bg-slate-950/40 rounded-xl border border-slate-800/60"
            />
          ))}
        </div>
      </section>
    </div>
  );
}
