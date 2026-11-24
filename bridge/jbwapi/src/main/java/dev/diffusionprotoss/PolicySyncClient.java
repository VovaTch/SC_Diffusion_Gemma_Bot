package dev.diffusionprotoss;

import com.google.gson.Gson;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Non-blocking HTTP sync client between the JBWAPI S0 frame loop and the host
 * FastAPI policy server (default: http://172.18.0.1:8080/tick).
 */
public final class PolicySyncClient {
    private static final int CONNECT_TIMEOUT_MS = 200;
    private static final int READ_TIMEOUT_MS = 300;

    private final String tickUrl;
    private final Gson gson = new Gson();
    private final ExecutorService executor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "policy-sync");
        t.setDaemon(true);
        return t;
    });
    private final AtomicBoolean inFlight = new AtomicBoolean(false);
    private final AtomicReference<Protocol.CommandBundleDto> latestBundle =
            new AtomicReference<>(new Protocol.CommandBundleDto());

    public PolicySyncClient(String baseUrl) {
        String trimmed = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.tickUrl = trimmed + "/tick";
    }

    public Protocol.CommandBundleDto latestCommand() {
        return latestBundle.get();
    }

    /**
     * Queue a snapshot POST if no request is already in flight. Never blocks BWAPI.
     */
    public void submitSnapshot(final Protocol.GameStateDto state) {
        if (!inFlight.compareAndSet(false, true)) {
            return;
        }
        executor.submit(() -> {
            try {
                String payload = gson.toJson(state);
                HttpURLConnection conn = (HttpURLConnection) new URL(tickUrl).openConnection();
                conn.setRequestMethod("POST");
                conn.setConnectTimeout(CONNECT_TIMEOUT_MS);
                conn.setReadTimeout(READ_TIMEOUT_MS);
                conn.setDoOutput(true);
                conn.setRequestProperty("Content-Type", "application/json");

                byte[] bytes = payload.getBytes(StandardCharsets.UTF_8);
                try (OutputStream os = conn.getOutputStream()) {
                    os.write(bytes);
                }
                if (conn.getResponseCode() == 200) {
                    StringBuilder sb = new StringBuilder();
                    try (BufferedReader br = new BufferedReader(
                            new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                        String line;
                        while ((line = br.readLine()) != null) {
                            sb.append(line);
                        }
                    }
                    Protocol.CommandBundleDto bundle =
                            gson.fromJson(sb.toString(), Protocol.CommandBundleDto.class);
                    if (bundle != null) {
                        latestBundle.set(bundle);
                    }
                }
                conn.disconnect();
            } catch (Exception ignored) {
                // Host server offline or busy: S0 keeps running on the last valid bundle.
            } finally {
                inFlight.set(false);
            }
        });
    }

    public void shutdown() {
        executor.shutdownNow();
    }
}
