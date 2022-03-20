import {
  Button,
  Card,
  CardActions,
  CardContent,
  Container,
  Grid,
  TextField,
  Typography,
} from "@mui/material";
import React, { useEffect } from "react";

interface UnitcellMetaProps {
  jsonCallback: any;
}

const UnitcellMeta = (props: UnitcellMetaProps) => {
  const [repX, setRepX] = React.useState(1);
  const [repY, setRepY] = React.useState(1);
  const [repZ, setRepZ] = React.useState(1);
  const [posX, setPosX] = React.useState(0);
  const [posY, setPosY] = React.useState(0);
  const [posZ, setPosZ] = React.useState(0);

  useEffect(() => {
    props.jsonCallback(
      {
        repetitions: {
          x: repX,
          y: repY,
          z: repZ
        },
        translation: {
          x: posX,
          y: posY,
          z: posZ
        }
      }
      , "unitcellMeta"
    );
  }, [repX, repY, repZ, posX, posY, posZ]);


  return (
    <Card sx={{}}>
      <CardContent>
        <Typography>UnitcellMeta</Typography>
        <Grid container sx={{ paddingTop: 2 }} rowSpacing={2}>
          <Grid item xs={4}>
            <TextField
              label="repX"
              defaultValue={repX}
              onChange={(e) => {
                setRepX(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="repY"
              defaultValue={repY}
              onChange={(e) => {
                setRepY(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="repZ"
              defaultValue={repZ}
              onChange={(e) => {
                setRepZ(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="posX"
              defaultValue={posX}
              onChange={(e) => {
                setPosX(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="posY"
              defaultValue={posY}
              onChange={(e) => {
                setPosY(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="posZ"
              defaultValue={posZ}
              onChange={(e) => {
                setPosZ(Number(e.target.value));
              }}
            />
          </Grid>
        </Grid>
      </CardContent>
      <CardActions></CardActions>
    </Card>
  );
};

export default UnitcellMeta;
