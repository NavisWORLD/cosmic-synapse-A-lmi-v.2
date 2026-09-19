package world.navis.lighttoken.ui;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import world.navis.lighttoken.model.TokenDetail;

public final class EmbeddingChartModel {
    private final String tokenId;
    private final List<ChartPoint> points;

    private EmbeddingChartModel(String tokenId, List<ChartPoint> points) {
        this.tokenId = tokenId;
        this.points = List.copyOf(points);
    }

    public static EmbeddingChartModel from(TokenDetail token) {
        Objects.requireNonNull(token, "token");
        float[] values = token.vectors().embedding();
        List<ChartPoint> points = new ArrayList<>(values.length);
        for (int index = 0; index < values.length; index++) {
            points.add(new ChartPoint(index, values[index]));
        }
        return new EmbeddingChartModel(token.summary().tokenId(), points);
    }

    public String tokenId() {
        return tokenId;
    }

    public List<ChartPoint> points() {
        return points;
    }
}
