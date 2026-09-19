package world.navis.lighttoken.ui;

import java.util.List;
import java.util.Objects;
import world.navis.lighttoken.model.QueryResult;
import world.navis.lighttoken.model.SearchHit;

public final class SearchResultsModel {
    private final QueryResult result;

    public SearchResultsModel(QueryResult result) {
        this.result = Objects.requireNonNull(result, "result");
    }

    public QueryResult result() {
        return result;
    }

    public List<SearchHit> hits() {
        return result.hits();
    }
}
