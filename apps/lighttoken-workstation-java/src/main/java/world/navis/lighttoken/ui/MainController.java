package world.navis.lighttoken.ui;

import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import world.navis.lighttoken.model.CollectionSnapshot;
import world.navis.lighttoken.model.QueryResult;
import world.navis.lighttoken.model.SimilarityMethod;
import world.navis.lighttoken.model.TokenDetail;

public final class MainController {
    private final ExplorerViewModel viewModel;

    public MainController(ExplorerViewModel viewModel) {
        this.viewModel = Objects.requireNonNull(viewModel, "viewModel");
    }

    public CompletableFuture<CollectionSnapshot> open(Path source) {
        return viewModel.openCollection(source);
    }

    public TokenDetail select(String tokenId) {
        return viewModel.selectToken(tokenId);
    }

    public CompletableFuture<QueryResult> search(
            String tokenId, SimilarityMethod method, int topK) {
        return viewModel.search(tokenId, method, topK);
    }

    public ExplorerViewModel viewModel() {
        return viewModel;
    }
}
