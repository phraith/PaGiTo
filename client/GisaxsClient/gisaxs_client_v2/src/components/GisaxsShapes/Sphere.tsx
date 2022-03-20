import {
  DeleteForever,
  ExpandLess,
  ExpandMore
} from "@mui/icons-material";
import {
  Button,
  Card,
  CardActions,
  CardContent,
  Collapse,
  FormControl,
  Grid,
  Typography,
} from "@mui/material";
import { useEffect, useState } from "react";
import ParameterWrapper from "./ParameterWrapper";

interface SphereProps {
  id: string;
  removeCallback: any;
  jsonCallback: any;
}

const Sphere = (props: SphereProps) => {
  const [collapsed, setCollapsed] = useState(true);

  const [rMean, setRMean] = useState(5);
  const [rStddev, setRStddev] = useState(0);
  const [posX, setPosX] = useState(0);
  const [posY, setPosY] = useState(0);
  const [posZ, setPosZ] = useState(0);
  const [refBeta, setRefBeta] = useState(2e-8);
  const [refDelta, setRefDelta] = useState(6e-6);

  const handleButtonClick = () => {
    setCollapsed(!collapsed);
  };

  const handleRemove = (event) => {
    props.removeCallback();
  };

  useEffect(() => {
    props.jsonCallback(
      {
        type: "sphere",
        radius: {
          mean: rMean,
          stddev: rStddev,
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
  }, [rStddev, rMean, posX, posY, posZ, refBeta, refDelta]);

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
              Sphere {collapsed ? `[${rMean}, ${rStddev}]` : ""}
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

export default Sphere;
