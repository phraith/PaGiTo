import Button from "@mui/material/Button"
import Card from "@mui/material/Card"
import CardActions from "@mui/material/CardActions"
import CardContent from "@mui/material/CardContent"
import Collapse from "@mui/material/Collapse"
import InputAdornment from "@mui/material/InputAdornment"
import Grid from "@mui/material/Grid"
import Typography from "@mui/material/Typography"
import TextField from "@mui/material/TextField"

import { useState, useEffect } from "react";
import {
  InstrumentationConfig,
  SetLocalStorageEntity,
} from "../Utility/DefaultConfigs";
import Box from "@mui/material/Box/Box"

interface InstrumentationProps {
  jsonCallback: any;
  initialResX: number;
  initialResY: number;
}

const Instrumentation = ({ initialResX = 0, initialResY = 0, ...props }) => {
  const [alphaI, setAlphaI] = useState(InstrumentationConfig.beam.alphai);
  const [photonEv, setPhotonEv] = useState(
    InstrumentationConfig.beam.photonEv
  );
  const [beamX, setBeamX] = useState(
    InstrumentationConfig.detector.beamImpact.x
  );
  const [beamY, setBeamY] = useState(
    InstrumentationConfig.detector.beamImpact.y
  );

  let resWidth = initialResX == 0 ? InstrumentationConfig.detector.resolution.width : initialResX
  let resHeight = initialResY == 0 ? InstrumentationConfig.detector.resolution.height : initialResY

  const [resX, setResX] = useState(
    resWidth
  );
  const [resY, setResY] = useState(
    resHeight
  );
  const [pixelsize, setPixelsize] = useState(
    InstrumentationConfig.detector.pixelsize
  );
  const [sampleDistance, setSampleDistance] = useState(
    InstrumentationConfig.detector.sampleDistance
  );

  const localStorageEntityName: string = "instrumentationConfig"
  const configFieldName: string = "instrumentation"


  useEffect(() => {
    let resWidth = initialResX == 0 ? InstrumentationConfig.detector.resolution.width : initialResX
    let resHeight = initialResY == 0 ? InstrumentationConfig.detector.resolution.height : initialResY
    setResX(resWidth)
    setResY(resHeight)
  }, [initialResX, initialResY]);

  useEffect(() => {
    let currentConfig = {
      beam: {
        alphai: alphaI,
        photonEv: photonEv,
      },
      detector: {
        pixelsize: pixelsize,
        resolution: {
          width: resX,
          height: resY,
        },
        sampleDistance: sampleDistance,
        beamImpact: {
          x: beamX,
          y: beamY,
        },
      },
    };

    SetLocalStorageEntity(
      currentConfig,
      InstrumentationConfig,
      localStorageEntityName
    );

    props.jsonCallback(currentConfig, configFieldName);
  }, [alphaI, photonEv, beamX, beamY, resX, resY, pixelsize, sampleDistance]);

  useEffect(() => {
    let data = localStorage.getItem(localStorageEntityName);
    if (data !== null) {
      let instrumentationConfig = JSON.parse(data);
      setAlphaI(instrumentationConfig.beam.alphai);
      setPhotonEv(instrumentationConfig.beam.photonEv);
      setBeamX(instrumentationConfig.detector.beamImpact.x);
      setBeamY(instrumentationConfig.detector.beamImpact.y);
      setResX(instrumentationConfig.detector.resolution.width);
      setResY(instrumentationConfig.detector.resolution.height);
      setPixelsize(instrumentationConfig.detector.pixelsize);
      setSampleDistance(instrumentationConfig.detector.sampleDistance);
    }
  }, []);

  return (
    <Box>
      <Card sx={{ height: "100%" }} >
        <CardContent>
          <Typography>Instrumentation</Typography>
          <Box display="flex" sx={{ flexDirection: "column" }}>
            <Box display="flex" sx={{ paddingBottom: 1 }}>
              <TextField
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">°</InputAdornment>
                  ),
                }}
                label="alphaI"
                value={alphaI}
                type="number"
                onChange={(e) => {
                  setAlphaI(Number(e.target.value));
                }}
              />
              <TextField
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">eV</InputAdornment>
                  ),
                }}
                label="photonEv"
                value={photonEv}
                type="number"
                onChange={(e) => {
                  setPhotonEv(Number(e.target.value));
                }}
              />
            </Box>
            <Box display="flex" sx={{ paddingBottom: 1 }}>
              <TextField
                label="beamX"
                value={beamX}
                type="number"
                onChange={(e) => {
                  setBeamX(Number(e.target.value));
                }}
              />
              <TextField
                label="beamY"
                value={beamY}
                type="number"
                onChange={(e) => {
                  setBeamY(Number(e.target.value));
                }}
              />

              <TextField
                disabled={initialResX != 0}
                label="resX"
                value={resX}
                type="number"
                onChange={(e) => {
                  setResX(Number(e.target.value));
                }}
              />
              <TextField
                disabled={initialResY != 0}
                label="resY"
                value={resY}
                type="number"
                onChange={(e) => {
                  setResY(Number(e.target.value));
                }}
              />
            </Box>
            <Box display="flex" sx={{ paddingBottom: 1 }}>
              <TextField
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">mm</InputAdornment>
                  ),
                }}
                label="pixelsize"
                value={pixelsize}
                type="number"
                onChange={(e) => {
                  setPixelsize(Number(e.target.value));
                }}
              />
              <TextField
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">mm</InputAdornment>
                  ),
                }}
                label="sampleDistance"
                value={sampleDistance}
                type="number"
                onChange={(e) => {
                  setSampleDistance(Number(e.target.value));
                }}
              />
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Instrumentation;
