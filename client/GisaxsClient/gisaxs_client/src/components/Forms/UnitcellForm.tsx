import JSONForm from "@rjsf/material-ui";
import type { JSONSchema7 } from "json-schema";
import type { UiSchema } from "@rjsf/core";
import CustomObjectFieldTemplate from "./CustomObjectFieldTemplate"

const uiSchema: UiSchema = {
    unitcell: {
        "ui:ObjectFieldTemplate": CustomObjectFieldTemplate,
        repetitions: {
            repetitionsInX: {
                "ui:widget": "range",
            },
            repetitionsInY: {
                "ui:widget": "range",
            },
            repetitionsInZ: {
                "ui:widget": "range",
            },
        },
        distances: {
            distOnX: {
                "ui:widget": "range",
            },
            distOnY: {
                "ui:widget": "range",
            },
            distOnZ: {
                "ui:widget": "range",
            },
        }
    }
};

const schema2: JSONSchema7 = {
    type: "object",
    properties: {
        unitcell: {
            $ref: "#/definitions/unitcell",
        }
    },
    definitions: {
        unitcell: {
            title: "General unitcell configuration",
            type: "object",
            properties: {
                repetitions: {
                    title: "Repetitions",
                    type: "object",
                    properties: {
                        repetitionsInX: {
                            type: "integer",
                            default: 1
                        },
                        repetitionsInY: {
                            type: "integer",
                            default: 1
                        },
                        repetitionsInZ: {
                            type: "integer",
                            default: 1
                        },
                    },
                },
                distances: {
                    title: "Distances",
                    type: "object",
                    properties: {
                        distOnX: {
                            type: "integer",
                            default: 0
                        },
                        distOnY: {
                            type: "integer",
                            default: 0
                        },
                        distOnZ: {
                            type: "integer",
                            default: 0
                        },
                    },
                },
            },
        },
    },
};


function UnitcellForm(props: { callback: Function, formData: any }) {
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

export default UnitcellForm;
