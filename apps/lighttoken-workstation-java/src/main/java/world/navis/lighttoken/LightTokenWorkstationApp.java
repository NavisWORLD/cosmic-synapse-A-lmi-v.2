package world.navis.lighttoken;

import java.nio.file.Path;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;
import world.navis.lighttoken.library.LibraryDatabase;
import world.navis.lighttoken.library.TokenRepository;
import world.navis.lighttoken.nativebridge.JniNativeEngine;
import world.navis.lighttoken.nativebridge.NativeEngine;
import world.navis.lighttoken.service.CollectionService;
import world.navis.lighttoken.service.SearchService;
import world.navis.lighttoken.service.ServiceExecutor;
import world.navis.lighttoken.ui.ExplorerViewModel;
import world.navis.lighttoken.ui.ResonanceExplorerView;

public final class LightTokenWorkstationApp extends Application {
    private NativeEngine engine;
    private LibraryDatabase database;
    private ServiceExecutor workers;
    private CollectionService collections;
    private SearchService searches;
    private ExplorerViewModel viewModel;

    public static void launchApp(String[] args) {
        Application.launch(LightTokenWorkstationApp.class, args);
    }

    @Override
    public void start(Stage stage) throws Exception {
        engine = JniNativeEngine.loadDefault();
        database = LibraryDatabase.open(defaultDataRoot().resolve("library.db"));
        workers = ServiceExecutor.create();
        collections = new CollectionService(engine, new TokenRepository(database), workers);
        searches = new SearchService(engine, new TokenRepository(database), workers);
        viewModel = new ExplorerViewModel(engine, collections, searches);

        ResonanceExplorerView view = new ResonanceExplorerView(viewModel);
        Scene scene = new Scene(view, 1440, 900);
        stage.setTitle("A-LMI LightToken Resonance Explorer");
        stage.setMinWidth(1024);
        stage.setMinHeight(700);
        stage.setScene(scene);
        stage.show();
    }

    private static Path defaultDataRoot() {
        String configured = System.getenv("LIGHTTOKEN_DATA_ROOT");
        if (configured != null && !configured.isBlank()) {
            return Path.of(configured).toAbsolutePath().normalize();
        }
        return Path.of(System.getProperty("user.home"), ".almi", "lighttoken")
                .toAbsolutePath()
                .normalize();
    }

    @Override
    public void stop() throws Exception {
        if (viewModel != null) {
            viewModel.close();
        }
        if (searches != null) {
            searches.close();
        }
        if (collections != null) {
            collections.close();
        }
        if (workers != null) {
            workers.close();
        }
        if (database != null) {
            database.close();
        }
        if (engine != null) {
            engine.close();
        }
    }
}
