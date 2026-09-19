package world.navis.lighttoken.service;

import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public final class ServiceExecutor implements AutoCloseable {
    private static final AtomicInteger THREAD_SEQUENCE = new AtomicInteger();
    private final ExecutorService executor;
    private final int workerCount;

    private ServiceExecutor(int workerCount) {
        this.workerCount = workerCount;
        this.executor =
                Executors.newFixedThreadPool(
                        workerCount,
                        runnable -> {
                            Thread thread =
                                    new Thread(
                                            runnable,
                                            "lighttoken-service-"
                                                    + THREAD_SEQUENCE.incrementAndGet());
                            thread.setDaemon(true);
                            return thread;
                        });
    }

    public static int workerCountForProcessors(int processors) {
        int safeProcessors = Math.max(1, processors);
        return Math.max(1, Math.min(4, safeProcessors / 2));
    }

    public static ServiceExecutor create() {
        return new ServiceExecutor(
                workerCountForProcessors(Runtime.getRuntime().availableProcessors()));
    }

    public int workerCount() {
        return workerCount;
    }

    public <T> CompletableFuture<T> submit(Callable<T> operation) {
        Objects.requireNonNull(operation, "operation");
        CompletableFuture<T> result = new CompletableFuture<>();
        AtomicReference<Future<?>> submitted = new AtomicReference<>();
        Future<?> future =
                executor.submit(
                        () -> {
                            if (result.isCancelled()) {
                                return;
                            }
                            try {
                                result.complete(operation.call());
                            } catch (Throwable error) {
                                result.completeExceptionally(error);
                            }
                        });
        submitted.set(future);
        result.whenComplete(
                (value, error) -> {
                    if (result.isCancelled()) {
                        Future<?> task = submitted.get();
                        if (task != null) {
                            task.cancel(true);
                        }
                    }
                });
        return result;
    }

    @Override
    public void close() {
        executor.shutdownNow();
    }
}
