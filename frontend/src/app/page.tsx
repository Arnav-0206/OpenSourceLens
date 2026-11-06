import HeroSection from "@/components/landing/HeroSection";
import FeaturesShowcase from "@/components/landing/FeaturesShowcase";
import BackendStatusBanner from "@/components/landing/BackendStatusBanner";

export default function HomePage() {
  return (
    <div className="space-y-12 pb-12">
      {/* Dynamic Hero Banner with Project Mission & Quick Navigation */}
      <HeroSection />

      {/* Live Backend Connectivity & Diagnostic Probe */}
      <BackendStatusBanner />

      {/* Core ML Architectures & Data Science Deliverables */}
      <FeaturesShowcase />
    </div>
  );
}
