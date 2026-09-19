package world.navis.lighttoken.ui;

import java.io.File;
import java.util.Objects;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ListCell;
import javafx.scene.control.ListView;
import javafx.scene.control.Separator;
import javafx.scene.control.Spinner;
import javafx.scene.control.SplitPane;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TextArea;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.DirectoryChooser;
import world.navis.lighttoken.model.QueryResult;
import world.navis.lighttoken.model.SearchHit;
import world.navis.lighttoken.model.SimilarityMethod;
import world.navis.lighttoken.model.TokenDetail;

public final class ResonanceExplorerView extends BorderPane {
    public static final String EMBEDDING_CHART_TITLE = "Embedding spectrum";
    public static final String EMBEDDING_X_AXIS_LABEL = "Embedding dimension";
    public static final String SPECTRAL_CHART_TITLE = "Spectral magnitude";
    public static final String SPECTRAL_X_AXIS_LABEL = "Spectral bin";

    private final MainController controller;
    private final ListView<String> tokenList = new ListView<>();
    private final ListView<SearchHit> resultList = new ListView<>();
    private final ComboBox<SimilarityMethod> methodBox =
            new ComboBox<>(FXCollections.observableArrayList(SimilarityMethod.values()));
    private final Spinner<Integer> topK = new Spinner<>(1, 100, 5);
    private final LineChart<Number, Number> embeddingChart;
    private final LineChart<Number, Number> spectrumChart;
    private final Label sourceLabel = new Label("No collection open");
    private final Label tokenLabel = new Label("No token selected");
    private final Label backendLabel = new Label("Backend: unavailable");
    private final Label statusLabel = new Label("Ready");
    private final TextArea tokenInspector = new TextArea();

    public ResonanceExplorerView(ExplorerViewModel viewModel) {
        this.controller = new MainController(Objects.requireNonNull(viewModel, "viewModel"));
        this.embeddingChart =
                createChart(
                        EMBEDDING_CHART_TITLE, EMBEDDING_X_AXIS_LABEL, "Embedding value");
        this.spectrumChart =
                createChart(SPECTRAL_CHART_TITLE, SPECTRAL_X_AXIS_LABEL, "Magnitude");
        buildLayout();
        wireActions();
        refreshFromViewModel();
    }

    private static LineChart<Number, Number> createChart(
            String title, String xAxisLabel, String yAxisLabel) {
        NumberAxis xAxis = new NumberAxis();
        xAxis.setLabel(xAxisLabel);
        xAxis.setForceZeroInRange(false);
        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel(yAxisLabel);
        yAxis.setForceZeroInRange(false);
        LineChart<Number, Number> chart = new LineChart<>(xAxis, yAxis);
        chart.setTitle(title);
        chart.setCreateSymbols(false);
        chart.setAnimated(false);
        chart.setLegendVisible(true);
        return chart;
    }

    private void buildLayout() {
        setPadding(new Insets(10));

        Button openButton = new Button("Open collection");
        openButton.setId("open-collection");
        VBox libraryPane =
                new VBox(
                        8,
                        new Label("LightToken library"),
                        openButton,
                        sourceLabel,
                        new Separator(),
                        tokenList);
        libraryPane.setPadding(new Insets(8));
        libraryPane.setPrefWidth(280);
        VBox.setVgrow(tokenList, Priority.ALWAYS);

        Tab embeddingTab = new Tab("Embedding", embeddingChart);
        embeddingTab.setClosable(false);
        Tab spectrumTab = new Tab("Spectrum", spectrumChart);
        spectrumTab.setClosable(false);
        TabPane charts = new TabPane(embeddingTab, spectrumTab);

        methodBox.getSelectionModel().select(SimilarityMethod.COSINE);
        Button searchButton = new Button("Search");
        searchButton.setId("run-search");
        HBox searchControls =
                new HBox(
                        8,
                        new Label("Method"),
                        methodBox,
                        new Label("Top K"),
                        topK,
                        searchButton);

        tokenInspector.setEditable(false);
        tokenInspector.setWrapText(true);
        tokenInspector.setPrefRowCount(8);

        resultList.setCellFactory(
                ignored ->
                        new ListCell<>() {
                            @Override
                            protected void updateItem(SearchHit item, boolean empty) {
                                super.updateItem(item, empty);
                                if (empty || item == null) {
                                    setText(null);
                                } else {
                                    setText(
                                            "#"
                                                    + item.rank()
                                                    + "  "
                                                    + item.tokenId()
                                                    + "  score="
                                                    + String.format("%.6f", item.score()));
                                }
                            }
                        });

        VBox detailPane =
                new VBox(
                        8,
                        new Label("Token inspector"),
                        tokenLabel,
                        tokenInspector,
                        new Separator(Orientation.HORIZONTAL),
                        new Label("Native similarity search"),
                        searchControls,
                        new Label("Results"),
                        resultList);
        detailPane.setPadding(new Insets(8));
        detailPane.setPrefWidth(390);
        VBox.setVgrow(resultList, Priority.ALWAYS);

        SplitPane split = new SplitPane(libraryPane, charts, detailPane);
        split.setDividerPositions(0.18, 0.72);
        setCenter(split);

        HBox status = new HBox(16, backendLabel, statusLabel);
        status.setPadding(new Insets(8, 2, 0, 2));
        setBottom(status);

        openButton.setOnAction(
                event -> {
                    DirectoryChooser chooser = new DirectoryChooser();
                    chooser.setTitle("Open LightToken collection");
                    File selected =
                            chooser.showDialog(
                                    getScene() == null ? null : getScene().getWindow());
                    if (selected != null) {
                        openCollection(selected.toPath());
                    }
                });
        searchButton.setOnAction(event -> runSearch());
    }

    private void wireActions() {
        tokenList
                .getSelectionModel()
                .selectedItemProperty()
                .addListener(
                        (observable, previous, tokenId) -> {
                            if (tokenId != null) {
                                controller.select(tokenId);
                                refreshSelectedToken();
                            }
                        });

        resultList
                .getSelectionModel()
                .selectedItemProperty()
                .addListener(
                        (observable, previous, hit) -> {
                            if (hit != null) {
                                overlayCandidate(hit.tokenId());
                            }
                        });
    }

    private void openCollection(java.nio.file.Path source) {
        statusLabel.setText("Opening " + source + " …");
        controller.open(source)
                .whenComplete(
                        (snapshot, error) ->
                                Platform.runLater(
                                        () -> {
                                            if (error != null) {
                                                statusLabel.setText(
                                                        "Open failed: " + rootMessage(error));
                                                return;
                                            }
                                            statusLabel.setText(
                                                    "Loaded " + snapshot.tokens().size() + " tokens");
                                            refreshFromViewModel();
                                        }));
    }

    private void runSearch() {
        TokenDetail selected = controller.viewModel().selectedToken();
        if (selected == null) {
            statusLabel.setText("Select a query token first");
            return;
        }
        SimilarityMethod method = methodBox.getValue();
        int requestedTopK = topK.getValue();
        statusLabel.setText("Searching with native backend …");
        controller.search(selected.summary().tokenId(), method, requestedTopK)
                .whenComplete(
                        (result, error) ->
                                Platform.runLater(
                                        () -> {
                                            if (error != null) {
                                                statusLabel.setText(
                                                        "Search failed: " + rootMessage(error));
                                                return;
                                            }
                                            showSearchResult(result);
                                            statusLabel.setText(
                                                    "Search complete: "
                                                            + result.hits().size()
                                                            + " hits");
                                        }));
    }

    public void refreshFromViewModel() {
        requireFxThread();
        var viewModel = controller.viewModel();
        var backend = viewModel.backendInfo();
        backendLabel.setText("Backend: " + backend.activeBackend());

        var snapshot = viewModel.snapshot();
        if (snapshot == null) {
            return;
        }
        sourceLabel.setText(snapshot.source().toString());
        String selectedId =
                viewModel.selectedToken() == null
                        ? null
                        : viewModel.selectedToken().summary().tokenId();
        tokenList.setItems(
                FXCollections.observableArrayList(
                        snapshot.tokens().stream()
                                .map(token -> token.summary().tokenId())
                                .toList()));
        if (selectedId != null) {
            tokenList.getSelectionModel().select(selectedId);
        } else if (!tokenList.getItems().isEmpty()) {
            tokenList.getSelectionModel().selectFirst();
        }
        refreshSelectedToken();

        QueryResult last = viewModel.lastQueryResult();
        if (last != null) {
            showSearchResult(last);
        }
    }

    private void refreshSelectedToken() {
        TokenDetail selected = controller.viewModel().selectedToken();
        if (selected == null) {
            return;
        }
        tokenLabel.setText(selected.summary().tokenId());
        tokenInspector.setText(
                "Source: "
                        + selected.summary().sourceUri()
                        + "\nModality: "
                        + selected.summary().modality()
                        + "\nRaw data ref: "
                        + selected.summary().rawDataRef()
                        + "\nEmbedding values: "
                        + selected.vectors().embedding().length
                        + "\nSpectral bins: "
                        + selected.vectors().spectralPower().length);
        showEmbedding(selected);
        showSpectrum(selected, null);
    }

    private void showEmbedding(TokenDetail token) {
        EmbeddingChartModel model = EmbeddingChartModel.from(token);
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName(model.tokenId());
        for (ChartPoint point : model.points()) {
            series.getData().add(new XYChart.Data<>(point.index(), point.value()));
        }
        embeddingChart.getData().setAll(series);
    }

    private void showSpectrum(TokenDetail query, TokenDetail candidate) {
        spectrumChart.getData().clear();
        spectrumChart.getData().add(series(SpectrumChartModel.from(query), "query"));
        if (candidate != null
                && !candidate.summary().tokenId().equals(query.summary().tokenId())) {
            spectrumChart
                    .getData()
                    .add(series(SpectrumChartModel.from(candidate), "candidate"));
        }
    }

    private static XYChart.Series<Number, Number> series(
            SpectrumChartModel model, String role) {
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName(role + ": " + model.tokenId());
        for (ChartPoint point : model.points()) {
            series.getData().add(new XYChart.Data<>(point.index(), point.value()));
        }
        return series;
    }

    private void showSearchResult(QueryResult result) {
        resultList.setItems(FXCollections.observableArrayList(result.hits()));
        backendLabel.setText("Backend: " + result.backend());
    }

    public void overlayCandidate(String tokenId) {
        requireFxThread();
        var snapshot = controller.viewModel().snapshot();
        TokenDetail query = controller.viewModel().selectedToken();
        if (snapshot == null || query == null) {
            return;
        }
        showSpectrum(query, snapshot.token(tokenId));
    }

    public int embeddingPointCount() {
        requireFxThread();
        return embeddingChart.getData().isEmpty()
                ? 0
                : embeddingChart.getData().getFirst().getData().size();
    }

    public int spectrumPointCount() {
        requireFxThread();
        return spectrumChart.getData().isEmpty()
                ? 0
                : spectrumChart.getData().getFirst().getData().size();
    }

    private static void requireFxThread() {
        if (!Platform.isFxApplicationThread()) {
            throw new IllegalStateException("JavaFX view access must occur on the application thread");
        }
    }

    private static String rootMessage(Throwable error) {
        Throwable current = error;
        while (current.getCause() != null) {
            current = current.getCause();
        }
        return current.getMessage() == null
                ? current.getClass().getSimpleName()
                : current.getMessage();
    }
}
