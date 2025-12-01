import logging
import threading
import time
from typing import Dict, List, Optional, Any
import grpc
from concurrent import futures

from rl_distributed.learner import ThreadSafeReplayBuffer, LearnerRPCService, Experience
from idl.python import actor_learner_pb2, actor_learner_pb2_grpc


class SimpleExperienceCollectorLearner:
    """A minimal learner that only collects experiences from actors"""

    def __init__(self,
                 buffer_capacity: int = 1000000,
                 buffer_save_path: Optional[str] = None,
                 auto_save_interval: int = 300):  # Auto-save every 5 minutes
        """
        Initialize the experience collector learner
        
        Args:
            buffer_capacity: Maximum number of experiences to store
            buffer_save_path: Path to save/load the buffer (optional)
            auto_save_interval: Interval in seconds for auto-saving buffer
        """
        # Initialize replay buffer
        self.replay_buffer = ThreadSafeReplayBuffer(capacity=buffer_capacity)

        # Track registered actors
        self.registered_actors = {}
        self.actors_lock = threading.RLock()

        # Statistics
        self.stats = {
            'total_experiences': 0,
            'experiences_by_actor': {},
            'last_update_time': {}
        }
        self.stats_lock = threading.RLock()

        # Buffer persistence
        self.buffer_save_path = buffer_save_path
        self.auto_save_interval = auto_save_interval
        self.auto_save_thread = None
        self.shutdown_event = threading.Event()

        # Load existing buffer if path provided
        if self.buffer_save_path:
            self.replay_buffer.load_buffer(self.buffer_save_path)
            logging.info(
                f"Loaded {len(self.replay_buffer)} experiences from {self.buffer_save_path}"
            )

        # Start auto-save thread if enabled
        if self.buffer_save_path and self.auto_save_interval > 0:
            self._start_auto_save()

    def register_actor(self,
                       actor_id: str,
                       metadata: Dict[str, Any] = None) -> bool:
        """
        Register a new actor
        
        Args:
            actor_id: Unique identifier for the actor
            metadata: Optional metadata about the actor
            
        Returns:
            True if registration successful
        """
        with self.actors_lock:
            if actor_id not in self.registered_actors:
                self.registered_actors[actor_id] = {
                    'metadata': metadata or {},
                    'registered_at': time.time()
                }

                with self.stats_lock:
                    self.stats['experiences_by_actor'][actor_id] = 0
                    self.stats['last_update_time'][actor_id] = time.time()

                logging.info(f"Registered actor: {actor_id}")
                return True
            else:
                logging.warning(f"Actor {actor_id} already registered")
                return True  # Return True anyway since actor exists

    def add_experiences(self, experiences: List['Experience'], actor_id: str):
        """
        Add experiences from an actor to the replay buffer
        
        Args:
            experiences: List of experiences to add
            actor_id: ID of the actor sending experiences
        """
        # Check if actor is registered
        with self.actors_lock:
            if actor_id not in self.registered_actors:
                logging.warning(
                    f"Received experiences from unregistered actor: {actor_id}"
                )
                # Auto-register the actor
                self.register_actor(actor_id)

        print(experiences[0])
        # Add experiences to buffer
        self.replay_buffer.push_batch(experiences)

        # Update statistics
        with self.stats_lock:
            self.stats['total_experiences'] += len(experiences)
            self.stats['experiences_by_actor'][actor_id] += len(experiences)
            self.stats['last_update_time'][actor_id] = time.time()

        logging.debug(
            f"Added {len(experiences)} experiences from actor {actor_id}")

    def get_model_weights_for_actor(self, actor_id: str) -> Optional[Dict]:
        """
        Get model weights for an actor (not implemented in collector-only mode)
        
        Args:
            actor_id: ID of the requesting actor
            
        Returns:
            None (no model updates in collector-only mode)
        """
        # This collector doesn't train or update models
        return None

    def get_statistics(self) -> Dict:
        """Get current statistics about collected experiences"""
        with self.stats_lock:
            stats_copy = self.stats.copy()
            stats_copy['buffer_size'] = len(self.replay_buffer)
            stats_copy['num_actors'] = len(self.registered_actors)
            return stats_copy

    def save_buffer(self, path: Optional[str] = None):
        """
        Save the replay buffer to disk
        
        Args:
            path: Path to save the buffer (uses default if not provided)
        """
        save_path = path or self.buffer_save_path
        if save_path:
            self.replay_buffer.save_buffer(save_path)
            logging.info(
                f"Saved {len(self.replay_buffer)} experiences to {save_path}")

    def _start_auto_save(self):
        """Start the auto-save background thread"""

        def auto_save_worker():
            while not self.shutdown_event.is_set():
                # Wait for interval or shutdown signal
                if self.shutdown_event.wait(self.auto_save_interval):
                    break

                # Save buffer
                try:
                    self.save_buffer()
                except Exception as e:
                    logging.error(f"Auto-save failed: {e}")

        self.auto_save_thread = threading.Thread(target=auto_save_worker,
                                                 daemon=True)
        self.auto_save_thread.start()
        logging.info(
            f"Started auto-save thread (interval: {self.auto_save_interval}s)")

    def shutdown(self):
        """Gracefully shutdown the learner"""
        logging.info("Shutting down learner...")

        # Signal shutdown
        self.shutdown_event.set()

        # Wait for auto-save thread to finish
        if self.auto_save_thread:
            self.auto_save_thread.join(timeout=5)

        # Final save
        if self.buffer_save_path:
            self.save_buffer()

        # Log final statistics
        stats = self.get_statistics()
        logging.info(f"Final statistics: {stats}")


def start_collector_server(port: int = 50049,
                           buffer_capacity: int = 50000000,
                           buffer_save_path: Optional[str] = None,
                           max_workers: int = 1000):
    """
    Start the gRPC server for the experience collector
    
    Args:
        port: Port to listen on
        buffer_capacity: Maximum buffer size
        buffer_save_path: Path to save/load buffer
        max_workers: Maximum number of concurrent RPC handlers
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create learner
    learner = SimpleExperienceCollectorLearner(
        buffer_capacity=buffer_capacity, buffer_save_path=buffer_save_path)

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    # Add the service
    service = LearnerRPCService(learner)
    actor_learner_pb2_grpc.add_LearnerServiceServicer_to_server(
        service, server)

    # Start server
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logging.info(f"Experience collector server started on port {port}")
    logging.info(f"Buffer capacity: {buffer_capacity}")
    if buffer_save_path:
        logging.info(f"Buffer save path: {buffer_save_path}")

    try:
        # Keep server running
        while True:
            time.sleep(60)  # Sleep for 1 minute

            # Log statistics periodically
            stats = learner.get_statistics()
            logging.info(
                f"Current stats - Total experiences: {stats['total_experiences']}, "
                f"Buffer size: {stats['buffer_size']}, "
                f"Active actors: {stats['num_actors']}")
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
    finally:
        # Graceful shutdown
        learner.shutdown()
        server.stop(grace=10)
        logging.info("Server stopped")
