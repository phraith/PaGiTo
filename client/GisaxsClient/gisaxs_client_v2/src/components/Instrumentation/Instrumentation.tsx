import {
  Card,
  CardActions,
  CardContent,
  Grid,
  InputAdornment,
  TextField,
  Typography,
} from "@mui/material";
import React, { useEffect } from "react";
import {
  InstrumentationConfig,
  SetLocalStorageEntity,
} from "../Utility/DefaultConfigs";

interface InstrumentationProps {
  jsonCallback: any;
}

const Instrumentation = (props: InstrumentationProps) => {
  const [alphaI, setAlphaI] = React.useState(InstrumentationConfig.beam.alphai);
  const [photonEv, setPhotonEv] = React.useState(
    InstrumentationConfig.beam.photonEv
  );
  const [beamX, setBeamX] = React.useState(
    InstrumentationConfig.detector.beamImpact.x
  );
  const [beamY, setBeamY] = React.useState(
    InstrumentationConfig.detector.beamImpact.y
  );
  const [resX, setResX] = React.useState(
    InstrumentationConfig.detector.resolution.width
  );
  const [resY, setResY] = React.useState(
    InstrumentationConfig.detector.resolution.height
  );
  const [pixelsize, setPixelsize] = React.useState(
    InstrumentationConfig.detector.pixelsize
  );
  const [sampleDistance, setSampleDistance] = React.useState(
    InstrumentationConfig.detector.sampleDistance
  );

  const localStorageEntityName : string = "instrumentationConfig"
  const configFieldName : string = "instrumentation"

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
    <Card sx={{}}>
      <CardContent>
        <Typography>Instrumentation</Typography>
        <Grid container sx={{ paddingTop: 2 }} rowSpacing={2}>
          <Grid item xs={6}>
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
          </Grid>
          <Grid item xs={6}>
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
          </Grid>
          <Grid item xs={3}>
            <TextField
              label="beamX"
              value={beamX}
              type="number"
              onChange={(e) => {
                setBeamX(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={3}>
            <TextField
              label="beamY"
              value={beamY}
              type="number"
              onChange={(e) => {
                setBeamY(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={3}>
            <TextField
              label="resX"
              value={resX}
              type="number"
              onChange={(e) => {
                setResX(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={3}>
            <TextField
              label="resY"
              value={resY}
              type="number"
              onChange={(e) => {
                setResY(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={6}>
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
          </Grid>
          <Grid item xs={6}>
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
          </Grid>
        </Grid>
      </CardContent>
      <CardActions></CardActions>
    </Card>
  );
};

export default Instrumentation;
