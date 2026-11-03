"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Boxes,
  Check,
  ChevronDown,
  Layers,
  Search,
  Star,
} from "lucide-react";

interface RepoItem {
  owner: string;
  name: string;
  repo_name: string;
  domain: string;
  stars: string;
}

const canonicalRepos: RepoItem[] = [
  {
    owner: "microsoft",
    name: "vscode",
    repo_name: "microsoft/vscode",
    domain: "Developer Tools",
    stars: "165k",
  },
  {
    owner: "pytorch",
    name: "pytorch",
    repo_name: "pytorch/pytorch",
    domain: "ML Framework",
    stars: "85k",
  },
  {
    owner: "kubernetes",
    name: "kubernetes",
    repo_name: "kubernetes/kubernetes",
    domain: "Cloud Orchestration",
    stars: "110k",
  },
  {
    owner: "huggingface",
    name: "transformers",
    repo_name: "huggingface/transformers",
    domain: "Foundation Models",
    stars: "135k",
  },
];

interface RepoSelectorProps {
  currentRepo?: string;
  className?: string;
}

export default function RepoSelector({
  currentRepo,
  className = "",
}: RepoSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const dropdownRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  const selectedRepo = canonicalRepos.find(
    (r) => r.repo_name.toLowerCase() === currentRepo?.toLowerCase()
  );

  const filteredRepos = canonicalRepos.filter(
    (r) =>
      r.repo_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      r.domain.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Close on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  // Close on Escape key
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    }

    if (isOpen) {
      document.addEventListener("keydown", handleKeyDown);
    }
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen]);

  function handleSelect(repo: RepoItem) {
    setIsOpen(false);
    setSearchTerm("");
    router.push(`/repos/${repo.owner}/${repo.name}`);
  }

  return (
    <div ref={dropdownRef} className={`relative inline-block text-left ${className}`}>
      {/* Trigger Button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-slate-900/90 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 rounded-xl transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        <Boxes className="w-3.5 h-3.5 text-indigo-400" />
        <span className="font-semibold text-slate-200">
          {selectedRepo ? selectedRepo.repo_name : "Select Repository"}
        </span>
        <ChevronDown
          className={`w-3.5 h-3.5 text-slate-400 transition-transform ${
            isOpen ? "rotate-180 text-indigo-400" : ""
          }`}
        />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute left-0 mt-2 w-72 origin-top-left rounded-2xl bg-slate-900 border border-slate-800 shadow-2xl shadow-black/80 z-50 p-2 space-y-2 backdrop-blur-xl">
          {/* Quick Filter Search Input */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Filter repositories..."
              className="w-full pl-8 pr-3 py-1.5 text-xs bg-slate-950/80 border border-slate-800 rounded-xl text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/60"
              autoFocus
            />
          </div>

          {/* Repo List */}
          <div className="max-h-60 overflow-y-auto space-y-1 py-1">
            {filteredRepos.length > 0 ? (
              filteredRepos.map((repo) => {
                const isSelected =
                  currentRepo?.toLowerCase() === repo.repo_name.toLowerCase();

                return (
                  <button
                    key={repo.repo_name}
                    type="button"
                    onClick={() => handleSelect(repo)}
                    className={`w-full flex items-center justify-between p-2 rounded-xl text-left transition-colors group ${
                      isSelected
                        ? "bg-indigo-600/15 border border-indigo-500/30 text-white"
                        : "hover:bg-slate-800/70 text-slate-300"
                    }`}
                  >
                    <div className="space-y-0.5">
                      <div className="text-xs font-bold flex items-center gap-1.5">
                        <span className="text-slate-400">{repo.owner}/</span>
                        <span className="text-white group-hover:text-indigo-300 transition-colors">
                          {repo.name}
                        </span>
                      </div>
                      <div className="text-[10px] text-slate-400 font-medium">
                        {repo.domain}
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="flex items-center gap-1 text-[11px] font-semibold text-slate-400">
                        <Star className="w-3 h-3 text-amber-400 fill-amber-400" />
                        {repo.stars}
                      </span>
                      {isSelected && (
                        <Check className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                      )}
                    </div>
                  </button>
                );
              })
            ) : (
              <div className="p-3 text-center text-xs text-slate-500">
                No matching repositories found.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
