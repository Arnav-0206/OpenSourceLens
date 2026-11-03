import HeroSection from "@/components/landing/HeroSection";
import EcosystemShowcase from "@/components/landing/EcosystemShowcase";
import GlobalMetricsGrid from "@/components/landing/GlobalMetricsGrid";
import FeaturesShowcase from "@/components/landing/FeaturesShowcase";
import BackendStatusBanner from "@/components/landing/BackendStatusBanner";

export default function HomePage() {
  return (
    <div className="space-y-12 pb-12">
      {/* Dynamic Hero Banner with Project Mission & High-Level KPIs */}
      <HeroSection />

      {/* Live Backend Connectivity & Diagnostic Probe */}
      <BackendStatusBanner />

      {/* Target Ecosystem Cards with Live Star & SLA Aggregates */}
      <EcosystemShowcase />

      {/* Global Macro Key Performance Indicators */}
      <GlobalMetricsGrid />

      {/* Core ML Architectures & Data Science Deliverables */}
      <FeaturesShowcase />
    </div>
  );
}
