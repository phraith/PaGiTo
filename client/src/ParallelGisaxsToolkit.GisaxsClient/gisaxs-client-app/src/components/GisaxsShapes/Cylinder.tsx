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
import { CylinderConfig } from "../Utility/DefaultConfigs"
import LocationParameterWrapper from "./LocationParameterWrapper"
import RefractionParameterWrapper from "./RefractionParameterWrapper"
import ShapeParameterWrapper from "./ShapeParameterWrapper"
import CylinderIcon from '../../assets/cylinder.png'

interface CylinderProps {
  id: string;
  removeCallback: any;
  initialConfig: any;
  jsonCallback: any;
  isSimulation: boolean;
}

const Cylinder = (props: CylinderProps) => {
  const [collapsed, setCollapsed] = useState(true);
  const [jsonData, setJsonData] = useState(CylinderConfig);

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
    props.jsonCallback(
      jsonData,
      props.id
    );
  }, [jsonData]);

  return (
    <Card key={props.id} sx={{}}>
      <CardContent>
        <Box display="flex" gap={2}>
          <Button size="small" sx={{ color: "text.primary" }} onClick={handleButtonClick}>
            {collapsed ? <ExpandMore /> : <ExpandLess />}
          </Button>
          <Button size="small" sx={{ color: "text.primary" }} onClick={handleRemove}>
            <DeleteForever />
          </Button>
          <Box component="img" sx={{ height: 25 }} src={CylinderIcon} />
          <Typography color="text.primary" gutterBottom>
            {collapsed ? `r: ${jsonData.radius.meanUpper} h: ${jsonData.height.meanUpper}` : ""}
          </Typography>
        </Box>
        <Collapse in={!collapsed}>
          <Box display="flex" gap={2} sx={{ flexDirection: "column" }}>
            <ShapeParameterWrapper isSimulation={props.isSimulation} initialParameterConfig={props.initialConfig.radius} jsonCallback={jsonCallback} parameterName="radius" />
            <ShapeParameterWrapper isSimulation={props.isSimulation} initialParameterConfig={props.initialConfig.radius} jsonCallback={jsonCallback} parameterName="height" />
            <RefractionParameterWrapper initialRefractionConfig={props.initialConfig.refraction} jsonCallback={jsonCallback} />
            <LocationParameterWrapper initialLocationsConfig={props.initialConfig.locations[0]} jsonCallback={jsonCallback} />
          </Box>
        </Collapse>
      </CardContent>
      <CardActions></CardActions>
    </Card>
  );
};

export default Cylinder;