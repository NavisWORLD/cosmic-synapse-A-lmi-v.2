package world.navis.lighttoken.nativebridge;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import world.navis.lighttoken.model.BackendInfo;
import world.navis.lighttoken.model.ComparisonResult;
import world.navis.lighttoken.model.SearchHit;
import world.navis.lighttoken.model.SearchRequest;
import world.navis.lighttoken.model.SearchResult;
import world.navis.lighttoken.model.SimilarityMethod;
import world.navis.lighttoken.model.TokenVectors;
import world.navis.lighttoken.model.ValidationResult;
import world.navis.lighttoken.model.WorkspaceReadResult;
import world.navis.lighttoken.model.WorkspaceTokenPayload;

public final class JniNativeEngine implements NativeEngine {
    public static final int EXPECTED_ABI_VERSION = 1;
    private static final ObjectMapper JSON = new ObjectMapper();

    private long context;
    private final int abiVersion;

    private JniNativeEngine(long context, int abiVersion) {
        this.context = context;
        this.abiVersion = abiVersion;
    }

    public static JniNativeEngine load(Path trustedNativeDirectory) {
        Objects.requireNonNull(trustedNativeDirectory, "trustedNativeDirectory");
        Path directory = trustedNativeDirectory.toAbsolutePath().normalize();
        if (!Files.isDirectory(directory) || Files.isSymbolicLink(directory)) {
            throw new IllegalArgumentException(
                    "native directory must be a real application-owned directory: " + directory);
        }

        Path library = directory.resolve(nativeLibraryName()).normalize();
        if (!Objects.equals(library.getParent(), directory)
                || !Files.isRegularFile(library)
                || Files.isSymbolicLink(library)) {
            throw new IllegalArgumentException(
                    "native library not found in trusted directory: " + library);
        }

        System.load(library.toString());
        int abi = nativeAbiVersion();
        if (abi != EXPECTED_ABI_VERSION) {
            throw new IllegalStateException(
                    "LightToken native ABI mismatch: expected "
                            + EXPECTED_ABI_VERSION
                            + ", got "
                            + abi);
        }
        long handle = nativeCreateContext();
        if (handle == 0L) {
            throw new IllegalStateException("LightToken native context creation failed");
        }
        return new JniNativeEngine(handle, abi);
    }

    public static JniNativeEngine loadDefault() {
        String configured = System.getProperty("lighttoken.native.dir", "").trim();
        if (!configured.isEmpty()) {
            return load(Path.of(configured));
        }
        try {
            Path codeLocation =
                    Path.of(
                                    JniNativeEngine.class
                                            .getProtectionDomain()
                                            .getCodeSource()
                                            .getLocation()
                                            .toURI())
                            .toAbsolutePath()
                            .normalize();
            Path base = Files.isDirectory(codeLocation) ? codeLocation : codeLocation.getParent();
            if (base == null) {
                throw new IllegalStateException("cannot resolve packaged application directory");
            }
            return load(base.resolve("native"));
        } catch (URISyntaxException error) {
            throw new IllegalStateException("cannot resolve packaged native directory", error);
        }
    }

    private static String nativeLibraryName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            return "lighttoken_ffi.dll";
        }
        if (os.contains("mac")) {
            return "liblighttoken_ffi.dylib";
        }
        return "liblighttoken_ffi.so";
    }

    private void requireOpen() {
        if (context == 0L) {
            throw new IllegalStateException("native context is closed");
        }
    }

    @Override
    public int abiVersion() {
        return abiVersion;
    }

    @Override
    public long contextHandle() {
        requireOpen();
        return context;
    }

    @Override
    public ValidationResult validateJson(String json) {
        requireOpen();
        JsonNode node = parse(nativeValidateJson(context, Objects.requireNonNull(json, "json")));
        return new ValidationResult(
                node.path("valid").asBoolean(),
                node.path("token_id").asText(),
                node.path("embedding_dimension").asInt(),
                node.path("spectral_dimension").asInt());
    }

    @Override
    public ComparisonResult compareJson(
            String leftJson, String rightJson, SimilarityMethod method) {
        requireOpen();
        Objects.requireNonNull(method, "method");
        JsonNode node =
                parse(
                        nativeCompareJson(
                                context,
                                Objects.requireNonNull(leftJson, "leftJson"),
                                Objects.requireNonNull(rightJson, "rightJson"),
                                method.wireName()));
        return new ComparisonResult(
                node.path("a_token_id").asText(),
                node.path("b_token_id").asText(),
                SimilarityMethod.fromWire(node.path("method").asText()),
                (float) node.path("score").asDouble(),
                node.path("backend").asText());
    }

    @Override
    public SearchResult searchJson(
            String queryJson, String collectionJson, SearchRequest request) {
        requireOpen();
        Objects.requireNonNull(request, "request");
        ObjectNode requestJson = JSON.createObjectNode();
        requestJson.put("method", request.method().wireName());
        if (request.topK() == null) {
            requestJson.putNull("top_k");
        } else {
            requestJson.put("top_k", request.topK());
        }
        if (request.threshold() == null) {
            requestJson.putNull("threshold");
        } else {
            requestJson.put("threshold", request.threshold());
        }
        if (request.modality() == null) {
            requestJson.putNull("modality");
        } else {
            requestJson.put("modality", request.modality());
        }
        if (request.sourcePrefix() == null) {
            requestJson.putNull("source_prefix");
        } else {
            requestJson.put("source_prefix", request.sourcePrefix());
        }

        JsonNode node =
                parse(
                        nativeSearchJson(
                                context,
                                Objects.requireNonNull(queryJson, "queryJson"),
                                Objects.requireNonNull(collectionJson, "collectionJson"),
                                requestJson.toString()));
        List<SearchHit> hits = new ArrayList<>();
        for (JsonNode hit : node.path("hits")) {
            hits.add(
                    new SearchHit(
                            hit.path("token_id").asText(),
                            (float) hit.path("score").asDouble(),
                            hit.path("rank").asInt()));
        }
        return new SearchResult(
                node.path("query_token_id").asText(),
                SimilarityMethod.fromWire(node.path("method").asText()),
                node.path("backend").asText(),
                hits);
    }

    @Override
    public TokenVectors vectorsJson(String json) {
        requireOpen();
        JsonNode node = parse(nativeVectorsJson(context, Objects.requireNonNull(json, "json")));
        return new TokenVectors(
                node.path("token_id").asText(),
                toFloatArray(node.path("embedding")),
                toFloatArray(node.path("spectral_power")));
    }

    @Override
    public WorkspaceReadResult readWorkspace(Path workspace) {
        requireOpen();
        Path source = requireSource(workspace, true);
        return parseWorkspace(nativeReadWorkspace(context, source.toString()));
    }

    @Override
    public WorkspaceReadResult readCosmos(Path bundle) {
        requireOpen();
        Path source = requireSource(bundle, false);
        return parseWorkspace(nativeReadCosmos(context, source.toString()));
    }

    private static Path requireSource(Path source, boolean directory) {
        Objects.requireNonNull(source, "source");
        Path normalized = source.toAbsolutePath().normalize();
        if (Files.isSymbolicLink(normalized)
                || (directory ? !Files.isDirectory(normalized) : !Files.isRegularFile(normalized))) {
            throw new IllegalArgumentException(
                    "invalid LightToken source: " + normalized);
        }
        return normalized;
    }

    private static WorkspaceReadResult parseWorkspace(String text) {
        JsonNode node = parse(text);
        List<WorkspaceTokenPayload> tokens = new ArrayList<>();
        for (JsonNode token : node.path("tokens")) {
            tokens.add(
                    new WorkspaceTokenPayload(
                            token.path("token_id").asText(),
                            token.path("source_path").asText(),
                            token.path("raw_data_ref").asText(),
                            nullableText(token, "canonical_json"),
                            token.path("resolvable").asBoolean(false),
                            token.path("verified").asBoolean(false),
                            token.path("raw_resolvable").asBoolean(false),
                            nullableText(token, "token_sha256"),
                            nullableText(token, "raw_sha256")));
        }
        return new WorkspaceReadResult(
                node.path("source_kind").asText(),
                node.path("bundle_verified").asBoolean(false),
                tokens);
    }

    private static String nullableText(JsonNode node, String field) {
        JsonNode value = node.get(field);
        return value == null || value.isNull() ? null : value.asText();
    }

    @Override
    public BackendInfo backendInfo() {
        requireOpen();
        JsonNode node = parse(nativeBackendJson(context));
        JsonNode cpp = node.path("cpp");
        return new BackendInfo(
                node.path("active").asText(),
                cpp.path("available").asBoolean(false),
                cpp.path("detail").asText(""));
    }

    private static float[] toFloatArray(JsonNode node) {
        float[] values = new float[node.size()];
        for (int index = 0; index < values.length; index++) {
            values[index] = (float) node.get(index).asDouble();
        }
        return values;
    }

    private static JsonNode parse(String text) {
        try {
            return JSON.readTree(text);
        } catch (IOException error) {
            throw new IllegalStateException("native engine returned invalid JSON", error);
        }
    }

    @Override
    public void close() {
        long handle = context;
        context = 0L;
        if (handle != 0L) {
            nativeFreeContext(handle);
        }
    }

    private static native int nativeAbiVersion();

    private static native long nativeCreateContext();

    private static native void nativeFreeContext(long context);

    private static native String nativeValidateJson(long context, String json);

    private static native String nativeCompareJson(
            long context, String leftJson, String rightJson, String method);

    private static native String nativeSearchJson(
            long context, String queryJson, String collectionJson, String requestJson);

    private static native String nativeVectorsJson(long context, String json);

    private static native String nativeReadWorkspace(long context, String path);

    private static native String nativeReadCosmos(long context, String path);

    private static native String nativeBackendJson(long context);
}
