import Button from "@mui/material/Button"
import Card from "@mui/material/Card"
import CardActions from "@mui/material/CardActions"
import CardContent from "@mui/material/CardContent"
import Collapse from "@mui/material/Collapse"
import FormControl from "@mui/material/FormControl"
import Grid from "@mui/material/Grid"
import Typography from "@mui/material/Typography"
import TextField from "@mui/material/TextField"
import Stack from "@mui/material/Stack"
import React, { useEffect } from "react";
import {
  SetLocalStorageEntity,
  UnitcellMetaConfig,
} from "../Utility/DefaultConfigs";

interface UnitcellMetaProps {
  jsonCallback: any;
}

const UnitcellMeta = (props: UnitcellMetaProps) => {
  const [repX, setRepX] = React.useState(UnitcellMetaConfig.repetitions.x);
  const [repY, setRepY] = React.useState(UnitcellMetaConfig.repetitions.y);
  const [repZ, setRepZ] = React.useState(UnitcellMetaConfig.repetitions.z);
  const [posX, setPosX] = React.useState(UnitcellMetaConfig.translation.x);
  const [posY, setPosY] = React.useState(UnitcellMetaConfig.translation.y);
  const [posZ, setPosZ] = React.useState(UnitcellMetaConfig.translation.z);

  const localStorageEntityName: string = "unitcellMetaConfig";
  const configFieldName: string = "unitcellMeta";

  useEffect(() => {
    let currentConfig = {
      repetitions: {
        x: repX,
        y: repY,
        z: repZ,
      },
      translation: {
        x: posX,
        y: posY,
        z: posZ,
      },
    };

    SetLocalStorageEntity(
      currentConfig,
      UnitcellMetaConfig,
      localStorageEntityName
    );

    props.jsonCallback(currentConfig, configFieldName);
  }, [repX, repY, repZ, posX, posY, posZ]);

  useEffect(() => {
    let data = localStorage.getItem(localStorageEntityName);
    if (data !== null) {
      let unitcellMetaConfig = JSON.parse(data);
      setRepX(unitcellMetaConfig.repetitions.x);
      setRepY(unitcellMetaConfig.repetitions.y);
      setRepZ(unitcellMetaConfig.repetitions.z);

      setPosX(unitcellMetaConfig.translation.x);
      setPosY(unitcellMetaConfig.translation.y);
      setPosZ(unitcellMetaConfig.translation.z);
    }
  }, []);

  return (
    <Card sx={{}}>
      <CardContent>
        <Typography>UnitcellMeta</Typography>
        <Grid container sx={{ paddingTop: 2 }} rowSpacing={2}>
          <Grid item xs={4}>
            <TextField
              label="repX"
              value={repX}
              type="number"
              onChange={(e) => {
                setRepX(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="repY"
              value={repY}
              type="number"
              onChange={(e) => {
                setRepY(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="repZ"
              value={repZ}
              type="number"
              onChange={(e) => {
                setRepZ(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="posX"
              value={posX}
              type="number"
              onChange={(e) => {
                setPosX(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="posY"
              value={posY}
              type="number"
              onChange={(e) => {
                setPosY(Number(e.target.value));
              }}
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="posZ"
              value={posZ}
              type="number"
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
