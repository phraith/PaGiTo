import * as _ from "lodash";

export const InstrumentationConfig = {
  beam: {
    alphai: 0.2,
    photonEv: 12398.4,
  },
  detector: {
    pixelsize: 57.3e-3,
    resolution: {
      width: 1475,
      height: 1679,
    },
    sampleDistance: 1000,
    beamImpact: {
      x: 737,
      y: 0,
    },
  },
};

export const UnitcellMetaConfig = {
  repetitions: {
    x: 1,
    y: 1,
    z: 1,
  },
  translation: {
    x: 0,
    y: 0,
    z: 0,
  },
};

export const SphereConfig = {
  radius: {
    mean: 5,
    stddev: 0,
  },
  refraction: {
    delta: 6e-6,
    beta: 2e-8,
  },
  locations: [
    {
      x: 0,
      y: 0,
      z: 0,
    },
  ],
};

export const CylinderConfig = {
  radius: {
    mean: 5,
    stddev: 0,
  },
  height: {
    mean: 5,
    stddev: 0,
  },
  refraction: {
    delta: 6e-6,
    beta: 2e-8,
  },
  locations: [
    {
      x: 0,
      y: 0,
      z: 0,
    },
  ],
};


export const SetLocalStorageEntity = (currentConfig: any, defaultConfig: any, entityName : string) => {
    if (!_.isEqual(currentConfig, defaultConfig)) {
        localStorage.setItem(
            entityName,
          JSON.stringify(currentConfig)
        );
      }
}