import { useState, useEffect, useMemo } from "react";
import Box from "@material-ui/core/Box";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Drawer from "@material-ui/core/Drawer";
import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import { makeStyles, styled } from "@material-ui/core/styles";
import ListItemIcon from "@material-ui/core/ListItemIcon";
import Divider from "@material-ui/core/Divider";
import ListItemText from "@material-ui/core/ListItemText";
import IconButton from "@material-ui/core/IconButton";
import InboxIcon from "@material-ui/icons/MoveToInbox";
import MailIcon from "@material-ui/icons/Mail";
import MenuIcon from "@material-ui/icons/Menu";
import Container from "@material-ui/core/Container";
import Typography from "@material-ui/core/Typography";
import Grid from "@material-ui/core/Grid";
import { debounce } from "lodash";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import Form from "../Forms/Form";
import UnitcellForm from "../Forms/UnitcellForm";
import InstrumentationForm from "../Forms/InstrumentationForm";
import ColormapForm from "../Forms/ColormapForm";
import PinchZoomPan from "./PanPinchZoom"

import {
  HubConnection,
  HubConnectionBuilder,
  HubConnectionState,
} from "@microsoft/signalr";

import "./styles.css";

const useStyles = makeStyles((theme) => ({
  root: {
    display: "flex",
  },
  appBar: {
    zIndex: theme.zIndex.drawer + 1,
  },
  title: {
    flexGrow: 1,
  },
  toolbar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "flex-end",
    padding: theme.spacing(0, 1),
    ...theme.mixins.toolbar,
  },
  content: {
    flexGrow: 1,
    padding: theme.spacing(3),
  },
  drawer: {
    width: 300,
    marginTop: "88px",
  },
}));

interface DataResponse {
  id: string;
  intensities: Int8Array;
}

function Content() {
  const classes = useStyles();
  const [formDataValue, setFormDataValue] = useState({});
  const [unitcellFormDataValue, setUnitcellFormDataValue] = useState({});
  const [colormapFormDataValue, setColormapFormDataValue] = useState({});


  
  const [instrumentationFormDataValue, setInstrumentationFormDataValue] =
    useState({});
  const [connection, setConnection] = useState<null | HubConnection>(null);
  const [intensities, setIntensities] = useState<string>();
  const [drawerState, setDrawerState] = useState<boolean>(false);
  useEffect(() => {
    const connect = new HubConnectionBuilder()
      .withUrl("/message")
      .withAutomaticReconnect()
      .build();

    connect
      .start()
      .then(() => {
        connect.on("ReceiveJobId", (message) => {
          receiveJobResult(message);
        });
      })
      .catch((error) => console.log(error));

    setConnection(connect);
  }, []);

  const receiveJobResult = (message: any) => {
    console.log(message)
    let url = "/api/redis?" + message;
    fetch(url)
      .then((response) => response.json())
      .then((data) => handleData(data));
  };

  const handleDrawerState = () => {
    console.log("drawerState");
    setDrawerState(!drawerState);
  };

  const toggleDrawer = (event: any) => {
    if (drawerState === true) {
      setDrawerState(false);
    }
  };

  const handleData = (input: any) => {
    var startTime = performance.now();
    setIntensities(input.data);
    var endTime = performance.now();
    console.log(`Handling data took ${endTime - startTime} milliseconds`);
  };

  const unitcellChangeHandler = (value: any) => {
    setUnitcellFormDataValue(value);
  };

  const debouncedUnitcellChangeHandler = useMemo(
    () => debounce(unitcellChangeHandler, 50, { trailing: true }),
    []
  );

  const instrumentationChangeHandler = (value: any) => {
    setInstrumentationFormDataValue(value);
  };

  const debouncedInstrumentationChangeHandler = useMemo(
    () => debounce(instrumentationChangeHandler, 50, { trailing: true }),
    []
  );

  const changeHandler = (value: any) => {
    setFormDataValue(value);
  };

  const debouncedChangeHandler = useMemo(
    () => debounce(changeHandler, 50, { trailing: true }),
    []
  );


  const colormapChangeHandler = (value: any) => {
    setColormapFormDataValue(value);
  };

  const debouncedColormapChangeHandler = useMemo(
    () => debounce(colormapChangeHandler, 50, { trailing: true }),
    []
  );

  useEffect(() => {
    const issueJob = async (message: any) => {
      if (connection?.state === HubConnectionState.Connected) {
        await connection.send("IssueJob", message);
      }
    };

    const getJobRequest = () => {
      console.log({ ...unitcellFormDataValue, ...formDataValue, ...colormapFormDataValue });

      let request = {
        info: {
          clientId: 0,
          jobId: 0,
          ...colormapFormDataValue
        },
        config: {
          ...instrumentationFormDataValue,
          ...unitcellFormDataValue,
          ...formDataValue,
        },
      };
      return JSON.stringify(request);
    };

    const sendRequest = async () => {
      if (connection?.state !== HubConnectionState.Connected) return;

      await issueJob(getJobRequest());
      console.log("after job sent");
    };

    (async () => {
      await sendRequest();
    })();
  }, [
    formDataValue,
    unitcellFormDataValue,
    colormapFormDataValue,
    instrumentationFormDataValue,
    connection,
  ]);

  const list = () => (
    <Box sx={{ width: 250 }} role="presentation">
      <List>
        {["Inbox", "Starred", "Send email", "Drafts"].map((text, index) => (
          <ListItem button key={text}>
            <ListItemIcon>
              {index % 2 === 0 ? <InboxIcon /> : <MailIcon />}
            </ListItemIcon>
            <ListItemText primary={text} />
          </ListItem>
        ))}
      </List>
      <Divider />
      <List>
        {["All mail", "Trash", "Spam"].map((text, index) => (
          <ListItem button key={text}>
            <ListItemIcon>
              {index % 2 === 0 ? <InboxIcon /> : <MailIcon />}
            </ListItemIcon>
            <ListItemText primary={text} />
          </ListItem>
        ))}
      </List>
    </Box>
  );
  
  const mouseMoveF = (e: any) => {
    console.log(e.screenX)
  }

  return (
    <Box onClick={toggleDrawer} onKeyDown={toggleDrawer}>
      <AppBar position="absolute" className={classes.appBar}>
        <IconButton />
        <Toolbar>
          <IconButton aria-label="menu" onClick={handleDrawerState}>
            <MenuIcon />
          </IconButton>
          <Typography>Dashboard</Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        anchor={"left"}
        open={drawerState}
        variant={"temporary"}
        BackdropProps={{ invisible: true }}
        classes={{
          paper: classes.drawer,
        }}
      >
        {list()}
      </Drawer>

      <Grid
        container
        direction="row"
        justifyContent="space-evenly"
        alignItems="center"
        spacing={3}
      >
        {/* <Grid item xs={8}>
            <PinchZoomPan width={1472} height={1679}>
              {(x: number, y : number, scale : number) => (
                  <img onMouseMove={mouseMoveF} alt="" src={`data:image/jpeg;base64,${intensities}`} style={{
                    pointerEvents: scale === 1 ? 'auto' : 'none',
                    transform: `translate3d(${x}px, ${y}px, 0) scale(${scale})`,
                    transformOrigin: '0 0',
                  }}/>
              )}
            </PinchZoomPan>
        </Grid> */}
        <Grid item xs={8}>
          <Container>
            <div>
            <TransformWrapper>
              <TransformComponent>
                <img onMouseMove={mouseMoveF} alt="" src={`data:image/jpeg;base64,${intensities}`}/>
              </TransformComponent>
            </TransformWrapper>
            </div>
          </Container>
        </Grid>

        <Grid item xs={4}>
          <Grid
            container
            direction="row"
            justifyContent="space-evenly"
            spacing={3}
            alignItems="flex-start"
          >
            <Grid item xs={6}>
              <InstrumentationForm
                callback={debouncedInstrumentationChangeHandler}
                formData={instrumentationFormDataValue}
              />
            </Grid>
            <Grid item xs={6}>
              <UnitcellForm
                callback={debouncedUnitcellChangeHandler}
                formData={unitcellFormDataValue}
              />
            </Grid>
            <Grid item xs={12} style={{ maxHeight: 800, overflow: "auto" }}>
              <Form
                callback={debouncedChangeHandler}
                formData={formDataValue}
              />
            </Grid>
            <Grid item xs={12}>
              <ColormapForm
                callback={colormapChangeHandler}
                formData={colormapFormDataValue}
              />
            </Grid>
          </Grid>
        </Grid>

        {/* <Grid item xs={6}>
          <Container>
            <TransformWrapper>
              <TransformComponent>
                <img alt="" src={`data:image/jpeg;base64,${intensities}`} />
              </TransformComponent>
            </TransformWrapper>
          </Container>
        </Grid>
        <Grid item xs={6}>
          <Container>
            <TransformWrapper>
              <TransformComponent>
                <img alt="" src={`data:image/jpeg;base64,${intensities}`} onMouseMove={mouseMoveF} />
              </TransformComponent>
            </TransformWrapper>
          </Container>
        </Grid> */}

      </Grid>
    </Box>
  );
}

export default Content;
