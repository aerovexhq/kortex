pub mod autotune;
pub mod avionics_bridge;
pub mod benchmark;
pub mod chat;
pub mod compression;
pub mod early_exit;
pub mod engine;
pub mod engines;
pub mod kb_rkhs;
pub mod kernels;
pub mod mesh;
pub mod model_cache;
pub mod sampling;
pub mod zero_adapt;
pub mod zo_specdec;

pub use autotune::{AutoTuner, BenchmarkProfile, ModelTuningReport, ScoreBreakdown};
pub use avionics_bridge::{
    AvionicsBridgeEvaluator, AvionicsBridgeReport, CanFdTelemetryFrame, CandidateActionToken,
    ChronosSeqlockArena, KestrelCanFdBridge, RigidBodyState6Dof, RolloutFeasibilityVerifier,
    RolloutSafetyConfig, SononAcousticLogitFrame, SononLogitRingBuffer, WatchdogState,
};
pub use benchmark::{BenchmarkReport, TokenSpeedCalculator};
pub use early_exit::{EarlyExitConfig, EarlyExitEvaluator, EarlyExitReport};
pub use engine::{EngineConfig, GenerationOutput, GenerationParams, KvPrecision, ModelEngine, ModelMemoryProfile, Utf8StreamDecoder};
pub use engines::{CpuEngine, HybridSinkEngine, QuantizedKvEngine, RawFfiEngine, SafeLlamaEngine, SlidingWindowEngine, SpeculativeEngine};
pub use kb_rkhs::{Complex64, KbRkhsConfig, KbRkhsEvaluator, KbRkhsHeadCache, KbRkhsReport};
pub use kernels::{
    AlignedTile64, BergmanSimdKernel, E8LatticeMicroKernel, MicroKernelBenchmarkReport,
    MicroKernelEvaluator, TiledContractionKernel,
};
pub use mesh::{
    CdrKvShardFrame, ConsistentHashRing, DistributedMeshEvaluator, DistributedMeshReport,
    PipelineStageAssignment, SwarmNode, SwarmPipelineScheduler,
};
pub use model_cache::{ModelCache, ModelCacheEntry, OptimalRunnerProfile, RunnerType};
pub use zero_adapt::{
    DeterministicGaussianRng, KestrelFlightReward, ZeroAdaptAdapter, ZeroAdaptConfig,
    ZeroAdaptEvaluator, ZeroAdaptReport,
};
pub use zo_specdec::{
    SpeculativeProjectionHead, SpeculativeStepOutcome, ZoSpecDecConfig, ZoSpecDecEngine,
    ZoSpecDecEvaluator, ZoSpecDecReport,
};

