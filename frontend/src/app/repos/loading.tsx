import { Skeleton, SkeletonCard } from "@/components/common/Skeleton";

export default function ReposLoading() {
  return (
    <div className="space-y-8 pb-12 animate-pulse">
      {/* Header & Search */}
      <div className="space-y-3">
        <Skeleton className="h-5 w-48 rounded-full" />
        <Skeleton className="h-10 w-80 max-w-full" />
        <Skeleton className="h-4 w-96 max-w-full" />
      </div>

      {/* Filter / Search input bar */}
      <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 flex items-center justify-between gap-4">
        <Skeleton className="h-10 w-72 rounded-xl" />
        <Skeleton className="h-10 w-44 rounded-xl" />
      </div>

      {/* 4 Repositories Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <SkeletonCard key={i} className="h-64" />
        ))}
      </div>
    </div>
  );
}
