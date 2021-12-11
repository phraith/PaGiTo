import JSONForm from "@rjsf/material-ui";
import type { JSONSchema7 } from "json-schema";
import type { UiSchema } from "@rjsf/core";
import CustomObjectFieldTemplate from "./CustomObjectFieldTemplate"
import CustomObjectFieldTemplateSmall from "./CustomObjectFieldTemplateSmall"

const uiSchema: UiSchema = {
    instrumentation: {
        scattering: {
            "ui:ObjectFieldTemplate": CustomObjectFieldTemplate,
        },
        detector: {
            "ui:ObjectFieldTemplate": CustomObjectFieldTemplateSmall,

            width: {
                "ui:widget": "range"
            },
            height: {
                "ui:widget": "range"
            },
            beamDirX: {
                "ui:widget": "range"
            },
            beamDirY: {
                "ui:widget": "range"
            }
        }

    }
};

const schema2: JSONSchema7 = {
    type: "object",
    properties: {
        instrumentation: {
            $ref: "#/definitions/instrumentation",
        }
    },
    definitions: {
        instrumentation: {
            title: "General instrumentation",
            type: "object",
            properties: {
                detector: {
                    title: "",
                    type: "object",
                    properties: {
                        width: {
                            type: "integer",
                            minimum: 1,
                            maximum: 2000,
                            default: 1475
                        },
                        height: {
                            type: "integer",
                            minimum: 1,
                            maximum: 2000,
                            default: 1679
                        },
                        beamDirX: {
                            type: "integer",
                            minimum: 0,
                            maximum: 2000,
                            default: 737
                        },
                        beamDirY: {
                            type: "integer",
                            minimum: 0,
                            maximum: 2000,
                            default: 0
                        },
                       
                    },
                },
                scattering: {
                    title: "",
                    type: "object",
                    properties: {
                        alphai: {
                            type: "number",
                            default: 0.2
                        },
                        beamev: {
                            type: "number",
                            default: 12398.4
                        },
                        pixelsize: {
                            type: "number",
                            default: 57.3e-3
                        },
                        detectorDistance: {
                            type: "integer",
                            default: 1000
                        }
                    },
                },
            },
        },
    },
};


function InstrumentationForm(props: { callback: Function, formData: any }) {
    const changeHandler = (value: any) => {
        props.callback(value.formData);
    };

    return (
            <JSONForm
                children={true}
                schema={schema2}
                formData={props.formData}
                uiSchema={uiSchema}
                onChange={(value) => { changeHandler(value) }}
                />
    );
}

export default InstrumentationForm;
