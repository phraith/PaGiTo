import DeleteForever from "@mui/icons-material/DeleteForever"
import ExpandLess from "@mui/icons-material/ExpandLess"
import ExpandMore from "@mui/icons-material/ExpandMore"
import Box from "@mui/material/Box/Box"
import Button from "@mui/material/Button"
import Card from "@mui/material/Card"
import CardContent from "@mui/material/CardContent"
import Collapse from "@mui/material/Collapse"
import Typography from "@mui/material/Typography"
import { useEffect, useState } from "react";
import { SphereConfig } from "../Utility/DefaultConfigs"
import LocationParameterWrapper from "./LocationParameterWrapper"
import RefractionParameterWrapper from "./RefractionParameterWrapper"
import ShapeParameterWrapper from "./ShapeParameterWrapper"

import SphereIcon from '../../assets/sphere.png'

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
    props.jsonCallback(
      jsonData,
      props.id
    );
  }, [jsonData]);

  return (
    <Card key={props.id}>
      <CardContent>
        <Box display="flex" gap={2}>
          <Button size="small" sx={{ color: "text.primary" }} onClick={handleButtonClick}>
            {collapsed ? <ExpandMore /> : <ExpandLess />}
          </Button>
          <Button size="small" sx={{ color: "text.primary" }} onClick={handleRemove}>
            <DeleteForever />
          </Button>
          <Box component="img" sx={{ height: 25 }} src={SphereIcon} />
          <Typography color="text.primary" gutterBottom>
            {collapsed ? `r: ${jsonData.radius.meanUpper}` : ""}
          </Typography>

        </Box>

        <Collapse in={!collapsed} sx={{ gap: "2" }}>
          <Box display="flex" gap={2} sx={{ flexDirection: "column" }}>
            <ShapeParameterWrapper isSimulation={props.isSimulation} initialParameterConfig={props.initialConfig.radius} jsonCallback={jsonCallback} parameterName="radius" />
            <RefractionParameterWrapper initialRefractionConfig={props.initialConfig.refraction} jsonCallback={jsonCallback} />
            <LocationParameterWrapper initialLocationsConfig={props.initialConfig.locations[0]} jsonCallback={jsonCallback} />
          </Box>
        </Collapse>
      </CardContent>
    </Card>
  );
};

export default Sphere;
