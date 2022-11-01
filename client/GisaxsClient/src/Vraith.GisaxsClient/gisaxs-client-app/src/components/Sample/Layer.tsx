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
import RefractionParameterWrapper from "../GisaxsShapes/RefractionParameterWrapper";
import { LayerConfig } from "../Utility/DefaultConfigs";
import Box from "@mui/material/Box/Box";

interface LayerProps {
  id: string;
  initialConfig: any;
  removeCallback?: any;
  jsonCallback: any;
}

const Layer = (props: LayerProps) => {
  const [thickness, setThickness] = useState(props.initialConfig.thickness);
  const [jsonData, setJsonData] = useState(props.initialConfig);

  useEffect(() => {
    props.jsonCallback(
      jsonData,
      props.id
    );
  }, [jsonData]);

  useEffect(() => {
    jsonCallback(thickness, "thickness")
  }, [thickness]);

  const jsonCallback = (value, key) => {
    jsonData[key] = value;
    setJsonData({ ...jsonData });
  };

  const [collapsed, setCollapsed] = useState(true);

  const handleButtonClick = () => {
    setCollapsed(!collapsed);
  };

  const handleRemove = (event) => {
    props.removeCallback();
  };

  const isSubstrate = props.removeCallback === undefined
  return (
    <Card sx={{}}>
      <CardContent>
        {!isSubstrate ?
          <Box display="flex" sx={{ justifyContent: "space-between" }}>
            <Button size="small" onClick={handleButtonClick}>
              {collapsed ? <ExpandMore /> : <ExpandLess />}
            </Button>

            <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>
              Layer
            </Typography>

            <Button size="small" onClick={handleRemove}>
              <DeleteForever />
            </Button>

          </Box>
          :
          <Box display="flex" sx={{ justifyContent: "space-between" }}>
            <Button size="small" onClick={handleButtonClick}>
              {collapsed ? <ExpandMore /> : <ExpandLess />}
            </Button>
            <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>
              Substrate
            </Typography>
          </Box>
        }
        <Collapse in={!collapsed}>
          <Box display="flex" sx={{ flexDirection: "column" }}>
            {!isSubstrate &&
              <Box display="flex" >
                <ParameterWrapper
                  defaultValue={thickness}
                  valueSetter={setThickness}
                  parameterName="thickness"
                />
              </Box>
            }
            <RefractionParameterWrapper initialRefractionConfig={props.initialConfig.refraction} jsonCallback={jsonCallback} />
          </Box>
        </Collapse>
      </CardContent>
    </Card>
  );
};

export default Layer;
