import Card from "@mui/material/Card"
import CardContent from "@mui/material/CardContent"
import Typography from "@mui/material/Typography"
import TextField from "@mui/material/TextField"
import { useState, useEffect } from "react";
import {
  SetLocalStorageEntity,
  UnitcellMetaConfig,
} from "../Utility/DefaultConfigs";
import Box from "@mui/material/Box/Box"

interface UnitcellMetaProps {
  jsonCallback: any;
}

const UnitcellMeta = (props: UnitcellMetaProps) => {
  const [repX, setRepX] = useState(UnitcellMetaConfig.repetitions.x);
  const [repY, setRepY] = useState(UnitcellMetaConfig.repetitions.y);
  const [repZ, setRepZ] = useState(UnitcellMetaConfig.repetitions.z);
  const [posX, setPosX] = useState(UnitcellMetaConfig.translation.x);
  const [posY, setPosY] = useState(UnitcellMetaConfig.translation.y);
  const [posZ, setPosZ] = useState(UnitcellMetaConfig.translation.z);

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
      <Card>
        <CardContent>
          <Box display="flex" gap={2} sx={{ flexDirection: "column" }}>
          <Typography>UnitcellMeta</Typography>
            <Box display="flex" gap={2}>
              <TextField
                label="repX"
                value={repX}
                type="number"
                onChange={(e) => {
                  setRepX(Number(e.target.value));
                }}
              />

              <TextField
                label="repY"
                value={repY}
                type="number"
                onChange={(e) => {
                  setRepY(Number(e.target.value));
                }}
              />

              <TextField
                label="repZ"
                value={repZ}
                type="number"
                onChange={(e) => {
                  setRepZ(Number(e.target.value));
                }}
              />
            </Box>
            <Box display="flex" gap={2}>
              <TextField
                label="posX"
                value={posX}
                type="number"
                onChange={(e) => {
                  setPosX(Number(e.target.value));
                }}
              />

              <TextField
                label="posY"
                value={posY}
                type="number"
                onChange={(e) => {
                  setPosY(Number(e.target.value));
                }}
              />

              <TextField
                label="posZ"
                value={posZ}
                type="number"
                onChange={(e) => {
                  setPosZ(Number(e.target.value));
                }}
              />
            </Box>
          </Box>
        </CardContent>
      </Card>
  );
};

export default UnitcellMeta;
