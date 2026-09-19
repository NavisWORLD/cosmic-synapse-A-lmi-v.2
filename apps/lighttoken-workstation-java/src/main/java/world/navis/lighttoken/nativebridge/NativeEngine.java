package world.navis.lighttoken.nativebridge;

import java.nio.file.Path;
import world.navis.lighttoken.model.BackendInfo;
import world.navis.lighttoken.model.ComparisonResult;
import world.navis.lighttoken.model.SearchRequest;
import world.navis.lighttoken.model.SearchResult;
import world.navis.lighttoken.model.SimilarityMethod;
import world.navis.lighttoken.model.TokenVectors;
import world.navis.lighttoken.model.ValidationResult;
import world.navis.lighttoken.model.WorkspaceReadResult;

public interface NativeEngine extends AutoCloseable {
    int abiVersion();

    long contextHandle();

    ValidationResult validateJson(String json);

    ComparisonResult compareJson(String leftJson, String rightJson, SimilarityMethod method);

    SearchResult searchJson(String queryJson, String collectionJson, SearchRequest request);

    TokenVectors vectorsJson(String json);

    WorkspaceReadResult readWorkspace(Path workspace);

    WorkspaceReadResult readCosmos(Path bundle);

    BackendInfo backendInfo();

    @Override
    void close();
}
