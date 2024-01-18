"""Defines the Fresnel Device."""
from typing import cast

import pulser

DEFAULT_NUMBER_OF_SHOTS = 2000

FresnelDevice = cast(
    pulser.devices._device_datacls.Device,
    pulser.json.abstract_repr.deserializer.deserialize_device(
        """
        {
            "version": "1",
            "channels":
            [
                {
                    "id": "rydberg_global",
                    "basis": "ground-rydberg",
                    "addressing": "Global",
                    "max_abs_detuning": 54.97787143782138,
                    "max_amp": 12.56637061435917,
                    "min_retarget_interval": null,
                    "fixed_retarget_t": null,
                    "max_targets": null,
                    "clock_period": 4,
                    "min_duration": 16,
                    "max_duration": 100000000,
                    "mod_bandwidth": 8,
                    "eom_config":
                        {
                            "limiting_beam": "RED",
                            "max_limiting_amp": 188.49555921538757,
                            "intermediate_detuning": 2827.4333882308138,
                            "controlled_beams": ["BLUE"],
                            "mod_bandwidth": 40,
                            "custom_buffer_time": 240
                    }
                }
            ],
            "name": "Fresnel",
            "dimensions": 2,
            "rydberg_level": 60,
            "min_atom_distance": 5,
            "max_atom_num": 25,
            "max_radial_distance": 35,
            "interaction_coeff_xy": null,
            "supports_slm_mask": false,
            "max_layout_filling": 0.5,
            "max_sequence_duration": 10000,
            "max_runs": 2000,
            "reusable_channels": false,
            "pre_calibrated_layouts":
            [
                {
                    "coordinates":
                        [
                            [-20.0, 0.0],
                            [-17.5, -4.330127],
                            [-17.5, 4.330127],
                            [-15.0, -8.660254],
                            [-15.0, 0.0],
                            [-15.0, 8.660254],
                            [-12.5, -12.990381],
                            [-12.5, -4.330127],
                            [-12.5, 4.330127],
                            [-12.5, 12.990381],
                            [-10.0, -17.320508],
                            [-10.0, -8.660254],
                            [-10.0, 0.0],
                            [-10.0, 8.660254],
                            [-10.0, 17.320508],
                            [-7.5, -12.990381],
                            [-7.5, -4.330127],
                            [-7.5, 4.330127],
                            [-7.5, 12.990381],
                            [-5.0, -17.320508],
                            [-5.0, -8.660254],
                            [-5.0, 0.0],
                            [-5.0, 8.660254],
                            [-5.0, 17.320508],
                            [-2.5, -12.990381],
                            [-2.5, -4.330127],
                            [-2.5, 4.330127],
                            [-2.5, 12.990381],
                            [0.0, -17.320508],
                            [0.0, -8.660254],
                            [0.0, 0.0],
                            [0.0, 8.660254],
                            [0.0, 17.320508],
                            [2.5, -12.990381],
                            [2.5, -4.330127],
                            [2.5, 4.330127],
                            [2.5, 12.990381],
                            [5.0, -17.320508],
                            [5.0, -8.660254],
                            [5.0, 0.0],
                            [5.0, 8.660254],
                            [5.0, 17.320508],
                            [7.5, -12.990381],
                            [7.5, -4.330127],
                            [7.5, 4.330127],
                            [7.5, 12.990381],
                            [10.0, -17.320508],
                            [10.0, -8.660254],
                            [10.0, 0.0],
                            [10.0, 8.660254],
                            [10.0, 17.320508],
                            [12.5, -12.990381],
                            [12.5, -4.330127],
                            [12.5, 4.330127],
                            [12.5, 12.990381],
                            [15.0, -8.660254],
                            [15.0, 0.0],
                            [15.0, 8.660254],
                            [17.5, -4.330127],
                            [17.5, 4.330127],
                            [20.0, 0.0]
                        ],
                    "slug": "TriangularLatticeLayout(61, 5.0\u00b5m)"
                }
            ],
            "is_virtual": false
        }
        """
    ),
)
