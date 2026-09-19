package world.navis.lighttoken;

import world.navis.lighttoken.nativebridge.JniNativeEngine;
import world.navis.lighttoken.nativebridge.NativeEngine;

public final class LightTokenWorkstation {
    private LightTokenWorkstation() {}

    public static void main(String[] args) {
        if (args.length == 1 && "--packaged-smoke".equals(args[0])) {
            try (NativeEngine engine = JniNativeEngine.loadDefault()) {
                if (engine.abiVersion() != JniNativeEngine.EXPECTED_ABI_VERSION) {
                    throw new IllegalStateException("unexpected native ABI");
                }
                System.out.println("LightToken packaged JNI smoke: " + engine.backendInfo().activeBackend());
                return;
            }
        }
        LightTokenWorkstationApp.launchApp(args);
    }
}
