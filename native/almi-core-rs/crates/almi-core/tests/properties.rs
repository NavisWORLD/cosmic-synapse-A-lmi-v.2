use almi_core::{AuthorityPolicy, ProviderProvenance, RoutingState};
use proptest::prelude::*;
use serde_json::Value;
use std::collections::BTreeMap;

proptest! {
    #[test]
    fn deny_authority_round_trip_preserves_all_domains(
        tools in proptest::collection::vec("[a-z0-9_-]{0,16}", 0..8),
        network in proptest::collection::vec("[a-z0-9_-]{0,16}", 0..8),
        filesystem in proptest::collection::vec("[a-z0-9_-]{0,16}", 0..8),
        cloud in proptest::collection::vec("[a-z0-9_-]{0,16}", 0..8),
        deployment in proptest::collection::vec("[a-z0-9_-]{0,16}", 0..8),
        actuator in proptest::collection::vec("[a-z0-9_-]{0,16}", 0..8),
    ) {
        let original = AuthorityPolicy {
            version: 1,
            tool_authority: tools,
            network_authority: network,
            filesystem_authority: filesystem,
            cloud_authority: cloud,
            deployment_authority: deployment,
            actuator_authority: actuator,
        };
        let encoded = serde_json::to_vec(&original).unwrap();
        let decoded: AuthorityPolicy = serde_json::from_slice(&encoded).unwrap();
        prop_assert_eq!(decoded, original);
    }

    #[test]
    fn routing_state_round_trip_preserves_order_independent_values(
        pairs in proptest::collection::vec(("[a-z]{1,12}", any::<i64>()), 0..20),
    ) {
        let routes: BTreeMap<String, Value> = pairs.into_iter()
            .map(|(key, value)| (key, Value::from(value)))
            .collect();
        let original = RoutingState { version: 1, routes };
        let encoded = serde_json::to_vec(&original).unwrap();
        let decoded: RoutingState = serde_json::from_slice(&encoded).unwrap();
        prop_assert_eq!(decoded, original);
    }

    #[test]
    fn provider_provenance_round_trip_never_invents_authority(
        provider in proptest::option::of("[a-z]{1,12}"),
        model in proptest::option::of("[a-z0-9_-]{1,16}"),
    ) {
        let original = ProviderProvenance {
            version: 1,
            provider_id: provider,
            model_id: model,
            revision: None,
            endpoint: None,
            capabilities: vec!["text".into()],
            context_limit: None,
            updated_at: None,
            last_response_provenance: None,
        };
        let encoded = serde_json::to_vec(&original).unwrap();
        let decoded: ProviderProvenance = serde_json::from_slice(&encoded).unwrap();
        prop_assert_eq!(decoded, original);
    }
}
