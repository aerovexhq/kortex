pub mod active_perception;
pub mod autotune;
pub mod avionics_bridge;
pub mod benchmark;
pub mod chat;
pub mod compression;
pub mod compressive_sensing;
pub mod dlpack_shm;
pub mod early_exit;
pub mod engine;
pub mod engines;
pub mod flocking_consensus;
pub mod graphon_attention;
pub mod kb_rkhs;
pub mod kernels;
pub mod lie_trajectory;
pub mod mesh;
pub mod model_cache;
pub mod multimodal_fusion;
pub mod neuromorphic_spike;
pub mod quantum_density;
pub mod sampling;
pub mod symplectic_dynamics;
pub mod tda_skeleton;
pub mod zero_adapt;
pub mod zo_specdec;

pub use active_perception::{
    angles_to_bin, bin_to_angles, great_circle_distance, spherical_to_cartesian,
    ActivePerceptionEvaluator, ActivePerceptionPlanner, ActivePerceptionReport,
    DirectionalEntropyField, GimbalOrientation, MinimumJerkTrajectory, SaccadeCommand,
    TrajectorySample, AZIMUTH_BINS, ELEVATION_BINS, TOTAL_DIRECTIONAL_BINS,
};
pub use autotune::{AutoTuner, BenchmarkProfile, ModelTuningReport, ScoreBreakdown};
pub use avionics_bridge::{
    AvionicsBridgeEvaluator, AvionicsBridgeReport, CanFdTelemetryFrame, CandidateActionToken,
    ChronosSeqlockArena, KestrelCanFdBridge, RigidBodyState6Dof, RolloutFeasibilityVerifier,
    RolloutSafetyConfig, SononAcousticLogitFrame, SononLogitRingBuffer, WatchdogState,
};
pub use benchmark::{BenchmarkReport, TokenSpeedCalculator};
pub use compressive_sensing::{
    CompressiveConfig, CompressiveKeyCache, CompressiveSensingEvaluator, CompressiveSensingReport,
    DctDictionary, DeterministicChirpOperator, FastCholeskyOmpSolver, CHIRP_PRIME,
    DEFAULT_HEAD_DIM, DEFAULT_MAX_SPARSITY, DEFAULT_MEASURE_DIM,
};
pub use dlpack_shm::{
    DlDataType, DlDataTypeCode, DlDevice, DlDeviceType, DlPackShmEvaluator, DlPackShmReport,
    DlTensorDescriptor, PosixShmRing, ShmError, ShmSlot, MAX_TENSOR_DIMS,
};
pub use early_exit::{EarlyExitConfig, EarlyExitEvaluator, EarlyExitReport};
pub use engine::{EngineConfig, GenerationOutput, GenerationParams, KvPrecision, ModelEngine, ModelMemoryProfile, Utf8StreamDecoder};
pub use engines::{CpuEngine, HybridSinkEngine, QuantizedKvEngine, RawFfiEngine, SafeLlamaEngine, SlidingWindowEngine, SpeculativeEngine};
pub use flocking_consensus::{
    DynamicFlockingTopology, FlockingAgentState, FlockingConsensusEvaluator, FlockingConsensusReport,
    ReynoldsFlockingController, SwarmAgentConfig,
};
pub use graphon_attention::{
    ContinuousGraphon, GraphonAttentionConfig, GraphonAttentionEvaluator,
    GraphonAttentionInterpolator, GraphonAttentionReport, SpectralGraphon,
    StepGraphonPartition, GRAPHON_EPSILON,
};
pub use kb_rkhs::{Complex64, KbRkhsConfig, KbRkhsEvaluator, KbRkhsHeadCache, KbRkhsReport};
pub use kernels::{
    AlignedTile64, BergmanSimdKernel, E8LatticeMicroKernel, MicroKernelBenchmarkReport,
    MicroKernelEvaluator, TiledContractionKernel,
};
pub use lie_trajectory::{
    exp_se3, log_se3, skew_symmetric, LieGroupPredictor, LieGroupPredictorEvaluator,
    LieGroupPredictorReport, Se3Pose, Se3Twist, TrajectoryLookaheadToken,
};
pub use mesh::{
    CdrKvShardFrame, ConsistentHashRing, DistributedMeshEvaluator, DistributedMeshReport,
    PipelineStageAssignment, SwarmNode, SwarmPipelineScheduler,
};
pub use model_cache::{ModelCache, ModelCacheEntry, OptimalRunnerProfile, RunnerType};
pub use multimodal_fusion::{
    AcousticObservation, CameraIntrinsics, FusedMultiModalToken, GatedCrossAttentionFusion,
    MultiModalFusionEvaluator, MultiModalFusionReport, VisualPatchObservation,
};
pub use neuromorphic_spike::{
    ActivePatchToken, DvsEvent, EventCameraConfig, EventSurfaceRaster, NeuromorphicAttentionBuffer,
    NeuromorphicSpikeEvaluator, NeuromorphicSpikeProjector, NeuromorphicSpikeReport,
    PatchEnergyTracker,
};
pub use quantum_density::{
    FactorizedDensityMatrix, PovmAttentionKernel, ProjectiveState, QuantumDensityConfig,
    QuantumDensityEvaluator, QuantumDensityReport, QuantumInformationEngine, NUMERICAL_EPSILON,
};
pub use symplectic_dynamics::{
    PhaseSpaceState, SeparableHamiltonian, ShadowHamiltonianTracker, StormerVerletIntegrator,
    SymplecticDynamicsEvaluator, SymplecticDynamicsReport, Yoshida4thIntegrator,
    DEFAULT_PHASE_DIM,
};
pub use tda_skeleton::{
    MapperAttentionGraph, MapperNode, MetricType, PersistenceDiagram, PersistencePair, Simplex1,
    SimplicialSkeletonPruner, TdaSkeletonEvaluator, TdaSkeletonReport, TopologicalKeyManifold,
    VietorisRipsFiltration,
};
pub use zero_adapt::{
    DeterministicGaussianRng, KestrelFlightReward, ZeroAdaptAdapter, ZeroAdaptConfig,
    ZeroAdaptEvaluator, ZeroAdaptReport,
};
pub use zo_specdec::{
    SpeculativeProjectionHead, SpeculativeStepOutcome, ZoSpecDecConfig, ZoSpecDecEngine,
    ZoSpecDecEvaluator, ZoSpecDecReport,
};

