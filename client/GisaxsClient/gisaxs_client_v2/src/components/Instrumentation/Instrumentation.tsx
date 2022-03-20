import {
  Card,
  CardActions,
  CardContent,
  Container,
  Grid,
  InputAdornment,
  TextField,
  Typography,
} from "@mui/material";
import React, { useEffect } from "react";

interface InstrumentationProps {
  jsonCallback: any;
}

const Instrumentation = (props: InstrumentationProps) => {
  const [alphaI, setAlphaI] = React.useState(0.2);
  const [photonEv, setPhotonEv] = React.useState(12398.4);
  const [beamX, setBeamX] = React.useState(737);
  const [beamY, setBeamY] = React.useState(0);
  const [resX, setResX] = React.useState(1475);
  const [resY, setResY] = React.useState(1679);
  const [pixelsize, setPixelsize] = React.useState(57.3e-3);
  const [sampleDistance, setSampleDistance] = React.useState(1000);

  useEffect(() => {
    props.jsonCallback(
      {
        beam: {
          alphai: alphaI,
          photonEv: photonEv
        },
        detector: {
          pixelsize: pixelsize,
          resolution: {
            width: resX,
            height: resY
          },
          sampleDistance: sampleDistance,
          beamImpact: {
            x: beamX,
            y: beamY
          }
        }
      }
      , "instrumentation"
    );
  }, [alphaI, photonEv, beamX, beamY, resX, resY, pixelsize, sampleDistance]);

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
              defaultValue={alphaI}
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
              defaultValue={photonEv}
              type="number"
              onChange={(e) => {
                setPhotonEv(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={3}>
            <TextField
              label="beamX"
              defaultValue={beamX}
              type="number"
              onChange={(e) => {
                setBeamX(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={3}>
            <TextField
              label="beamY"
              defaultValue={beamY}
              type="number"
              onChange={(e) => {
                setBeamY(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={3}>
            <TextField
              label="resX"
              defaultValue={resX}
              type="number"
              onChange={(e) => {
                setResX(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={3}>
            <TextField
              label="resY"
              defaultValue={resY}
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
              defaultValue={pixelsize}
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
              defaultValue={sampleDistance}
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
