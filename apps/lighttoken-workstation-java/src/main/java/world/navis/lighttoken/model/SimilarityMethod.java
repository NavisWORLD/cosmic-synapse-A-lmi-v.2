package world.navis.lighttoken.model;

public enum SimilarityMethod {
    POWER_CORRELATION("power_correlation"),
    COSINE("cosine"),
    EUCLIDEAN("euclidean");

    private final String wireName;

    SimilarityMethod(String wireName) {
        this.wireName = wireName;
    }

    public String wireName() {
        return wireName;
    }

    public static SimilarityMethod fromWire(String value) {
        for (SimilarityMethod method : values()) {
            if (method.wireName.equals(value)) {
                return method;
            }
        }
        throw new IllegalArgumentException("unsupported similarity method: " + value);
    }
}
