package world.navis.lighttoken.model;

public record SearchFilters(String modality, String sourcePrefix) {
    public static SearchFilters none() {
        return new SearchFilters(null, null);
    }
}
