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
  type: "sphere",
  radius: {
    meanUpper: 5,
    meanLower: 5,
    stddevUpper: 0,
    stddevLower: 0
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
  type: "cylinder",
  radius: {
    meanUpper: 5,
    meanLower: 5,
    stddevUpper: 0,
    stddevLower: 0
  },
  height: {
    meanUpper: 5,
    meanLower: 5,
    stddevUpper: 0,
    stddevLower: 0
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

export const LayerConfig = {
  refraction: {
    delta: 6e-6,
    beta: 2e-8,
  },
  thickness: 0.01
};

const isEqualsJson = (obj1,obj2)=>{
  let keys1 = Object.keys(obj1);
  let keys2 = Object.keys(obj2);
  return keys1.length === keys2.length && Object.keys(obj1).every(key=>obj1[key]==obj2[key]);
}

export const SetLocalStorageEntity = (
  currentConfig: any,
  defaultConfig: any,
  entityName: string
) => {
  if (!isEqualsJson(currentConfig, defaultConfig)) {
    localStorage.setItem(entityName, JSON.stringify(currentConfig));
  }
};
