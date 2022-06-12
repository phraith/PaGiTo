import DeleteForever from "@mui/icons-material/DeleteForever"
import ExpandLess from "@mui/icons-material/ExpandLess"
import ExpandMore from "@mui/icons-material/ExpandMore"
import Collapse from "@mui/material/Collapse";
import Button from "@mui/material/Button"
import Card from "@mui/material/Card"
import CardActions from "@mui/material/CardActions"
import CardContent from "@mui/material/CardContent"
import FormControl from "@mui/material/FormControl"
import TextField from "@mui/material/TextField"
import Grid from "@mui/material/Grid"
import Typography from "@mui/material/Typography"
import List from "@mui/material/List"
import ListItem from "@mui/material/ListItem"
import { useEffect, useState } from "react";
import ParameterWrapper from "../GisaxsShapes/ParameterWrapper";

interface LayerProps {
  id: string;
  order: number;
  removeCallback: any;
  initialConfig: any;
  jsonCallback: any;
}

const Layer = (props: LayerProps) => {
  const [collapsed, setCollapsed] = useState(true);

  const [thickness, setThickness] = useState(props.initialConfig.thickness);
  const [refBeta, setRefBeta] = useState(props.initialConfig.refraction.beta);
  const [refDelta, setRefDelta] = useState(props.initialConfig.refraction.delta);

  const handleButtonClick = () => {
    setCollapsed(!collapsed);
  };

  const handleRemove = (event) => {
    props.removeCallback();
  };

  useEffect(() => {
    props.jsonCallback(
      {
        refraction: {
          delta: refDelta,
          beta: refBeta,
        },
        order: props.order,
        thickness: thickness
      },
      props.id
    );
  }, [thickness, refBeta, refDelta]);

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
              Layer{" "}
              {collapsed
                ? `[${props.order}, ${thickness.toExponential()}, ${refBeta.toExponential()}, ${refDelta.toExponential()}]`
                : ""}
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
                <TextField
                  label="order"
                  type="number"
                  variant="outlined"
                  inputProps={{
                    readOnly: true,
                    disabled: true
                  }}
                  value={props.order}
                />
              </Grid>
              <Grid item xs={6}>
              <ParameterWrapper
                  defaultValue={thickness}
                  valueSetter={setThickness}
                  parameterName="thickness"
                />
              </Grid>
              <Grid item xs={6}>
              <ParameterWrapper
                  defaultValue={refBeta}
                  valueSetter={setRefBeta}
                  parameterName="beta"
                />
              </Grid>
              <Grid item xs={6}>
              <ParameterWrapper
                  defaultValue={refDelta}
                  valueSetter={setRefDelta}
                  parameterName="delta"
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

export default Layer;
