import DeleteForever from "@mui/icons-material/DeleteForever"
import ExpandLess from "@mui/icons-material/ExpandLess"
import ExpandMore from "@mui/icons-material/ExpandMore"
import Box from "@mui/material/Box/Box"
import Button from "@mui/material/Button"
import Card from "@mui/material/Card"
import CardActions from "@mui/material/CardActions"
import CardContent from "@mui/material/CardContent"
import Collapse from "@mui/material/Collapse"
import FormControl from "@mui/material/FormControl"
import Grid from "@mui/material/Grid"
import Typography from "@mui/material/Typography"
import { useEffect, useState } from "react";
import { SphereConfig } from "../Utility/DefaultConfigs"
import LocationParameterWrapper from "./LocationParameterWrapper"
import RefractionParameterWrapper from "./RefractionParameterWrapper"
import ShapeParameterWrapper from "./ShapeParameterWrapper"

interface SphereProps {
  id: string;
  initialConfig: any;
  removeCallback: any;
  jsonCallback: any;
  isSimulation: boolean;
}

const Sphere = (props: SphereProps) => {
  const [collapsed, setCollapsed] = useState(true);
  const [jsonData, setJsonData] = useState(SphereConfig);

  const handleButtonClick = () => {
    setCollapsed(!collapsed);
  };

  const handleRemove = (event) => {
    props.removeCallback();
  };

  const jsonCallback = (value, key) => {
    jsonData[key] = value;
    setJsonData({ ...jsonData });
  };

  useEffect(() => {
    console.log(jsonData)
    props.jsonCallback(
      jsonData,
      props.id
    );
  }, [jsonData]);

  return (
    <Card key={props.id} sx={{}}>
      <CardContent>
        <Grid container sx={{ paddingBottom: collapsed ? 0 : 2 }}>
          <Grid item xs={12}>
            <Box display="flex" sx={{ justifyContent: "space-between" }}>
              <Button size="small" onClick={handleButtonClick}>
                {collapsed ? <ExpandMore /> : <ExpandLess />}
              </Button>
              <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>
                Sphere {collapsed ? `[${jsonData.radius.meanUpper}, ${jsonData.radius.stddevUpper}]` : ""}
              </Typography>
              <Button size="small" onClick={handleRemove}>
                <DeleteForever />
              </Button>
            </Box>
          </Grid>
        </Grid>
        <Collapse in={!collapsed}>
          <FormControl>
            <Grid container direction={"row"} rowSpacing={1}>
              <Grid item xs={12}>
                  <ShapeParameterWrapper isSimulation={props.isSimulation} initialParameterConfig={props.initialConfig.radius} jsonCallback={jsonCallback} parameterName="radius" />
                  <RefractionParameterWrapper initialRefractionConfig={props.initialConfig.refraction} jsonCallback={jsonCallback} />
                  <LocationParameterWrapper initialLocationsConfig={props.initialConfig.locations[0]} jsonCallback={jsonCallback} />
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