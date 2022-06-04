import DeleteForever from "@mui/icons-material/DeleteForever"
import ExpandLess from "@mui/icons-material/ExpandLess"
import ExpandMore from "@mui/icons-material/ExpandMore"

import Button from "@mui/material/Button"
import Card from "@mui/material/Card"
import CardActions from "@mui/material/CardActions"
import CardContent from "@mui/material/CardContent"
import Collapse from "@mui/material/Collapse"
import FormControl from "@mui/material/FormControl"
import Grid from "@mui/material/Grid"
import Typography from "@mui/material/Typography"

import { useEffect, useState } from "react";
import ParameterWrapper from "./ParameterWrapper";

interface CylinderProps {
  id: string;
  removeCallback: any;
  initialConfig: any;
  jsonCallback: any;
}

const Cylinder = (props: CylinderProps) => {
  const [rMean, setRMean] = useState(props.initialConfig.radius.mean);
  const [rStddev, setRStddev] = useState(props.initialConfig.radius.stddev);
  const [hMean, setHMean] = useState(props.initialConfig.height.mean);
  const [hStddev, setHStddev] = useState(props.initialConfig.height.stddev);
  //fix locations
  const [posX, setPosX] = useState(props.initialConfig.locations[0].x);
  const [posY, setPosY] = useState(props.initialConfig.locations[0].y);
  const [posZ, setPosZ] = useState(props.initialConfig.locations[0].z);
  const [refBeta, setRefBeta] = useState(props.initialConfig.refraction.beta);
  const [refDelta, setRefDelta] = useState(props.initialConfig.refraction.delta);

  const [collapsed, setCollapsed] = useState(true);

  const handleButtonClick = () => {
    setCollapsed(!collapsed);
  };

  const handleRemove = (event) => {
    props.removeCallback();
  };

  useEffect(() => {
    props.jsonCallback(
      {
        type: "cylinder",
        radius: {
          mean: rMean,
          stddev: rStddev,
        },
        height: {
          mean: hMean,
          stddev: hStddev,
        },
        refraction: {
          delta: refDelta,
          beta: refBeta,
        },
        locations: [
          {
            x: posX,
            y: posY,
            z: posZ,
          },
        ],
      },
      props.id
    );
  }, [rStddev, rMean, hStddev, hMean,posX, posY, posZ, refBeta, refDelta]);

  return (
    <Card key={props.id} sx={{}}>
      <CardContent>
        <Grid
          container
          sx={{
            paddingBottom: collapsed ? 0 : 2,
          }}
        >
          <Grid item xs={6}>
            <Typography
              sx={{ fontSize: 14 }}
              color="text.secondary"
              gutterBottom
            >
              Cylinder {collapsed ? `[${rMean}, ${rStddev}]` : ""}
            </Typography>
          </Grid>
          <Grid item xs={3}>
            <Button size="small" onClick={handleButtonClick}>
              {collapsed ? <ExpandMore /> : <ExpandLess />}
            </Button>
          </Grid>
          <Grid item xs={3}>
            <Button size="small" onClick={handleRemove}>
              <DeleteForever />
            </Button>
          </Grid>
        </Grid>

        <Collapse in={!collapsed}>
          <FormControl>
            <Grid container direction={"row"} rowSpacing={1}>
              <Grid item xs={6}>
                <ParameterWrapper
                  defaultValue={rMean}
                  valueSetter={setRMean}
                  parameterName="radiusMean"
                />
              </Grid>
              <Grid item xs={6}>
                <ParameterWrapper
                  defaultValue={rStddev}
                  valueSetter={setRStddev}
                  parameterName="radiusStddev"
                />
              </Grid>
              <Grid item xs={6}>
                <ParameterWrapper
                  defaultValue={hMean}
                  valueSetter={setHMean}
                  parameterName="heightMean"
                />
              </Grid>
              <Grid item xs={6}>
                <ParameterWrapper
                  defaultValue={hStddev}
                  valueSetter={setHStddev}
                  parameterName="heightStddev"
                />
              </Grid>
              <Grid item xs={6}>
                <ParameterWrapper
                  defaultValue={refDelta}
                  valueSetter={setRefDelta}
                  parameterName="refDelta"
                />
              </Grid>
              <Grid item xs={6}>
                <ParameterWrapper
                  defaultValue={refBeta}
                  valueSetter={setRefBeta}
                  parameterName="refBeta"
                />
              </Grid>
              <Grid item xs={4}>
                <ParameterWrapper
                  defaultValue={posX}
                  valueSetter={setPosX}
                  parameterName="posX"
                />
              </Grid>
              <Grid item xs={4}>
                <ParameterWrapper
                  defaultValue={posY}
                  valueSetter={setPosY}
                  parameterName="posY"
                />
              </Grid>
              <Grid item xs={4}>
                <ParameterWrapper
                  defaultValue={posZ}
                  valueSetter={setPosZ}
                  parameterName="posZ"
                />
              </Grid>
            </Grid>
          </FormControl>
        </Collapse>
      </CardContent>
      <CardActions></CardActions>
    </Card>
  );
};

export default Cylinder;
